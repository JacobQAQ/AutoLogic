"""
LogicRAG query processing.

Given a user query q, this module embeds the query in the same vector space as
the state index produced by document_learner.py, selects states satisfying
sim(I(q), e(s_j)) > tau, and extracts a complete subtree from the global
template JSON so that all matched states are covered.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from document_learner import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingProvider,
    collect_materials,
    cosine_similarity,
    infer_node_type,
    node_sort_key,
    save_json,
)


DEFAULT_TAU = 0.5


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def get_template_nodes(template: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes = template.get("node_template", {}).get("nodes", [])
    if not nodes:
        nodes = template.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("Template JSON does not contain node_template.nodes.")
    return nodes


def get_index_states(index: Dict[str, Any]) -> List[Dict[str, Any]]:
    states = index.get("states", [])
    if not isinstance(states, list):
        raise ValueError("State index JSON does not contain a valid states list.")
    return states


def make_query_embedder(index: Dict[str, Any], args: argparse.Namespace) -> EmbeddingProvider:
    backend = index.get("embedding_backend", "")
    dimension = int(index.get("embedding_dimension") or 0)

    if backend == "local-hash" or args.local_embedding_only:
        return EmbeddingProvider(
            model=args.embedding_model,
            base_url=args.embedding_base_url,
            local_dim=dimension or 384,
            allow_api=False,
            batch_size=args.embedding_batch_size,
        )

    return EmbeddingProvider(
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        allow_api=True,
        batch_size=args.embedding_batch_size,
    )


def embed_query(query: str, index: Dict[str, Any], args: argparse.Namespace) -> List[float]:
    embedder = make_query_embedder(index, args)
    vector = embedder.embed_texts([query])[0]
    expected_dim = int(index.get("embedding_dimension") or 0)
    if expected_dim and len(vector) != expected_dim:
        raise ValueError(
            f"Query embedding dimension {len(vector)} does not match state index dimension {expected_dim}."
        )
    return vector


def rank_states_by_similarity(
    query_vector: Sequence[float],
    index: Dict[str, Any],
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for state in get_index_states(index):
        state_vector = state.get("embedding", [])
        sim = cosine_similarity(query_vector, state_vector)
        ranked.append(
            {
                "state_id": state.get("state_id") or state.get("node_id"),
                "node_id": state.get("node_id") or state.get("state_id"),
                "label": state.get("label", ""),
                "desc": state.get("desc", ""),
                "similarity": sim,
            }
        )
    ranked.sort(key=lambda item: item["similarity"], reverse=True)
    return ranked


def select_matched_states(
    ranked_states: Sequence[Dict[str, Any]],
    tau: float,
    fallback_top_k: int = 0,
) -> List[Dict[str, Any]]:
    matched = [state for state in ranked_states if state["similarity"] > tau]
    if matched or fallback_top_k <= 0:
        return matched
    return list(ranked_states[:fallback_top_k])


def build_node_maps(nodes: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    node_by_id = {str(node["node_id"]): node for node in nodes}
    children_by_id: Dict[str, List[str]] = {node_id: [] for node_id in node_by_id}

    for node in nodes:
        node_id = str(node["node_id"])
        children = [str(child) for child in node.get("children", []) if str(child) in node_by_id]
        children_by_id[node_id].extend(children)

    for node in nodes:
        node_id = str(node["node_id"])
        parent = node.get("parent")
        if parent is not None and str(parent) in node_by_id:
            parent_id = str(parent)
            if node_id not in children_by_id[parent_id]:
                children_by_id[parent_id].append(node_id)

    for node_id in children_by_id:
        children_by_id[node_id] = sorted(set(children_by_id[node_id]), key=node_sort_key)
    return node_by_id, children_by_id


def ancestor_chain(node_id: str, node_by_id: Dict[str, Dict[str, Any]]) -> List[str]:
    chain: List[str] = []
    current: Optional[str] = node_id
    visited = set()
    while current and current in node_by_id and current not in visited:
        visited.add(current)
        chain.append(current)
        parent = node_by_id[current].get("parent")
        current = str(parent) if parent is not None else None
    return chain


def lowest_common_ancestor(node_ids: Sequence[str], node_by_id: Dict[str, Dict[str, Any]]) -> str:
    valid_node_ids = [node_id for node_id in node_ids if node_id in node_by_id]
    if not valid_node_ids:
        raise ValueError("None of the matched states exist in the global template tree.")

    chains = [ancestor_chain(node_id, node_by_id) for node_id in valid_node_ids]
    common = set(chains[0])
    for chain in chains[1:]:
        common &= set(chain)

    if not common:
        raise ValueError("Matched states do not share a common ancestor in the template tree.")

    def depth(node_id: str) -> int:
        return len(ancestor_chain(node_id, node_by_id))

    return max(common, key=depth)


def collect_subtree_ids(root_id: str, children_by_id: Dict[str, List[str]]) -> List[str]:
    ordered: List[str] = []

    def visit(node_id: str) -> None:
        ordered.append(node_id)
        for child_id in children_by_id.get(node_id, []):
            visit(child_id)

    visit(root_id)
    return ordered


def subset_transitions(template: Dict[str, Any], subtree_ids: Sequence[str]) -> List[Dict[str, Any]]:
    subtree_set = set(subtree_ids)
    transitions = template.get("structure_pattern", {}).get("transitions", [])
    if not isinstance(transitions, list):
        return []

    kept = []
    for edge in transitions:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source in subtree_set and target in subtree_set:
            kept.append(deepcopy(edge))
    return kept


def extract_query_subtree(
    template: Dict[str, Any],
    matched_states: Sequence[Dict[str, Any]],
    query: str,
    tau: float,
    ranked_preview: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    nodes = get_template_nodes(template)
    node_by_id, children_by_id = build_node_maps(nodes)
    matched_node_ids = [str(state["node_id"]) for state in matched_states]
    root_id = lowest_common_ancestor(matched_node_ids, node_by_id)
    subtree_ids = collect_subtree_ids(root_id, children_by_id)
    subtree_set = set(subtree_ids)

    subtree_nodes: List[Dict[str, Any]] = []
    for node_id in subtree_ids:
        node = deepcopy(node_by_id[node_id])
        node["children"] = [child for child in children_by_id.get(node_id, []) if child in subtree_set]
        if node_id == root_id:
            node["parent"] = None
        elif node.get("parent") is not None:
            node["parent"] = str(node["parent"])
        node["node_type"] = infer_node_type(node.get("parent"), node["children"])
        subtree_nodes.append(node)

    subtree_nodes.sort(key=lambda node: node_sort_key(str(node["node_id"])))
    transitions = subset_transitions(template, subtree_ids)

    result = deepcopy(template)
    result["template_id"] = f"{template.get('template_id', 'logicrag')}_query_subtree"
    result["node_template"] = {"nodes": subtree_nodes}
    result["material_requirements_summary"] = collect_materials(subtree_nodes)
    result.setdefault("structure_pattern", {})
    result["structure_pattern"]["transitions"] = transitions
    result["query_processing_metadata"] = {
        "query": query,
        "tau": tau,
        "matched_state_count": len(matched_states),
        "matched_states": [
            {
                "node_id": state["node_id"],
                "label": state.get("label", ""),
                "similarity": round(float(state["similarity"]), 6),
            }
            for state in matched_states
        ],
        "subtree_root": root_id,
        "subtree_state_count": len(subtree_nodes),
        "top_ranked_preview": [
            {
                "node_id": state["node_id"],
                "label": state.get("label", ""),
                "similarity": round(float(state["similarity"]), 6),
            }
            for state in ranked_preview
        ],
    }
    return result


def run_query_processing(args: argparse.Namespace) -> Dict[str, Any]:
    template = load_json(args.template)
    index = load_json(args.index)

    query_vector = embed_query(args.query, index, args)
    ranked = rank_states_by_similarity(query_vector, index)
    matched = select_matched_states(ranked, args.tau, args.fallback_top_k)

    if not matched:
        preview = "\n".join(
            f"- {item['node_id']} sim={item['similarity']:.4f} label={item.get('label', '')}"
            for item in ranked[:5]
        )
        raise ValueError(
            f"No state satisfies sim(I(q), e(s_j)) > {args.tau}. "
            f"Top candidates are:\n{preview}"
        )

    subtree = extract_query_subtree(
        template=template,
        matched_states=matched,
        query=args.query,
        tau=args.tau,
        ranked_preview=ranked[: args.preview_top_k],
    )
    save_json(subtree, args.output)
    return subtree


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LogicRAG query processing")
    parser.add_argument("--query", required=True, help="User query q.")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Similarity threshold.")
    parser.add_argument("--template", default="logicrag_outputs/global_template.json", help="Global template JSON.")
    parser.add_argument("--index", default="logicrag_outputs/state_index.json", help="State embedding index JSON.")
    parser.add_argument("--output", default="logicrag_outputs/query_subtree.json", help="Output query-specific subtree JSON.")
    parser.add_argument("--preview-top-k", type=int, default=5, help="Number of ranked states saved in metadata.")
    parser.add_argument(
        "--fallback-top-k",
        type=int,
        default=0,
        help="Optional fallback when no state exceeds tau. Keep 0 for strict threshold behavior.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name.")
    parser.add_argument("--embedding-base-url", default=DEFAULT_EMBEDDING_BASE_URL, help="Embedding API base URL.")
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE, help="Embedding API batch size.")
    parser.add_argument(
        "--local-embedding-only",
        action="store_true",
        help="Use local hash embeddings instead of the embedding API.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    subtree = run_query_processing(args)
    meta = subtree["query_processing_metadata"]
    print("LogicRAG query processing completed.")
    print(f"- query: {meta['query']}")
    print(f"- matched states: {meta['matched_state_count']}")
    print(f"- subtree root: {meta['subtree_root']}")
    print(f"- subtree states: {meta['subtree_state_count']}")
    print(f"- output: {args.output}")


if __name__ == "__main__":
    main()
