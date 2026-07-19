from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "autologic_client.py"


def keyless_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "LOGICRAG_CHAT_API_KEY",
        "DASHSCOPE_API_KEY",
        "LOGICRAG_EMBEDDING_API_KEY",
        "IFIND_USERNAME",
        "IFIND_PASSWORD",
    ):
        env.pop(name, None)
    return env


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *arguments],
        cwd=PROJECT_ROOT,
        env=keyless_environment(),
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


class AutoLogicCLITests(unittest.TestCase):
    def test_top_level_and_subcommand_help(self) -> None:
        top = run_cli("--help")
        self.assertEqual(top.returncode, 0, top.stderr)
        for command in ("demo", "build", "generate", "run"):
            result = run_cli(command, "--help")
            self.assertEqual(result.returncode, 0, result.stderr)

    def _run_demo(self, scenario: str, output_root: Path) -> dict[str, object]:
        result = run_cli(
            "demo",
            "--scenario",
            scenario,
            "--output-root",
            str(output_root),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = {
            "dfa.json",
            "generated_report.md",
            "generated_states.json",
            "execution_trace.json",
            "run_manifest.json",
        }
        self.assertTrue(all((output_root / name).exists() for name in expected))
        return {
            "states": json.loads((output_root / "generated_states.json").read_text(encoding="utf-8")),
            "trace": json.loads((output_root / "execution_trace.json").read_text(encoding="utf-8")),
            "manifest": json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8")),
            "report": (output_root / "generated_report.md").read_text(encoding="utf-8"),
        }

    def test_demo_scenarios_are_keyless_and_take_different_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            up = self._run_demo("price-up", root / "up")
            down = self._run_demo("price-down", root / "down")
        self.assertEqual(up["manifest"]["executed_state_path"], ["S001", "S002", "S003", "S005"])
        self.assertEqual(down["manifest"]["executed_state_path"], ["S001", "S002", "S004", "S005"])
        self.assertNotEqual(up["manifest"]["executed_state_path"], down["manifest"]["executed_state_path"])

    def test_trace_symbols_are_finite_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self._run_demo("price-up", Path(directory))
        selected = []
        for step in payload["trace"]:
            symbol = step.get("selected_condition")
            if symbol is None:
                continue
            candidates = {item["symbol"] for item in step.get("candidate_conditions", [])}
            self.assertIn(symbol, candidates)
            selected.append(symbol)
        self.assertIn("PRICE_UP", selected)

    def test_generated_report_matches_visited_state_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self._run_demo("price-down", Path(directory))
        expected = "\n\n".join(
            item["generated_content"]
            for item in payload["states"]
            if item.get("generated_content")
        ).strip()
        self.assertEqual(payload["report"], expected)
        self.assertIn("bearish_analysis", payload["report"])
        self.assertNotIn("bullish_analysis", payload["report"])

    def test_invalid_dfa_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "invalid.json"
            invalid.write_text("{}", encoding="utf-8")
            result = run_cli(
                "generate",
                "--dfa",
                str(invalid),
                "--query",
                "write report for 2026-01-02",
                "--output-root",
                str(root / "output"),
                "--dry-run",
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AutoLogic error", result.stderr)

    def test_demo_direct_entry_does_not_open_network(self) -> None:
        import autologic_client

        with tempfile.TemporaryDirectory() as directory, patch(
            "socket.create_connection", side_effect=AssertionError("network access attempted")
        ):
            code = autologic_client.main(
                ["demo", "--scenario", "price-up", "--output-root", directory]
            )
        self.assertEqual(code, 0)

    def test_documented_commands_match_real_cli(self) -> None:
        document = (PROJECT_ROOT / "docs" / "AUTOLOGIC_DEMO.md").read_text(encoding="utf-8")
        for command in ("demo", "build", "generate", "run"):
            self.assertIn(f"python autologic_client.py {command}", document)
            self.assertEqual(run_cli(command, "--help").returncode, 0)

    def test_offline_build_generate_and_run_commands_are_executable(self) -> None:
        def sequence(document_id: str) -> dict[str, object]:
            return {
                "document_id": document_id,
                "state_sequence": [
                    {
                        "node_id": "1",
                        "node_type": "leaf",
                        "template_description": "Opening",
                        "level": 0,
                        "parent": None,
                        "children": [],
                        "content_guideline": "Write the opening.",
                        "required_materials": [],
                        "length": 50,
                    },
                    {
                        "node_id": "2",
                        "node_type": "leaf",
                        "template_description": "Risk warning",
                        "level": 0,
                        "parent": None,
                        "children": [],
                        "content_guideline": "Write the risk warning.",
                        "required_materials": [],
                        "length": 50,
                    },
                ],
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seq_a = root / "a.json"
            seq_b = root / "b.json"
            seq_a.write_text(json.dumps(sequence("report-a")), encoding="utf-8")
            seq_b.write_text(json.dumps(sequence("report-b")), encoding="utf-8")
            build_root = root / "build"
            built = run_cli(
                "build",
                "--from-state-sequences",
                str(seq_a),
                str(seq_b),
                "--local-embedding-only",
                "--output-root",
                str(build_root),
            )
            self.assertEqual(built.returncode, 0, built.stderr)
            self.assertTrue((build_root / "dfa.json").exists())
            self.assertTrue((build_root / "build_manifest.json").exists())

            generate_root = root / "generate"
            generated = run_cli(
                "generate",
                "--dfa",
                str(build_root / "dfa.json"),
                "--query",
                "write the weekly report for 2026-01-02",
                "--output-root",
                str(generate_root),
                "--dry-run",
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            self.assertTrue(all((generate_root / name).exists() for name in {
                "dfa.json", "generated_report.md", "generated_states.json",
                "execution_trace.json", "run_manifest.json",
            }))

            run_root = root / "run"
            full = run_cli(
                "run",
                "--from-state-sequences",
                str(seq_a),
                str(seq_b),
                "--local-embedding-only",
                "--query",
                "write the weekly report for 2026-01-02",
                "--output-root",
                str(run_root),
                "--dry-run",
            )
            self.assertEqual(full.returncode, 0, full.stderr)
            self.assertTrue((run_root / "build" / "build_manifest.json").exists())
            self.assertTrue((run_root / "run_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
