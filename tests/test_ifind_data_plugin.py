from __future__ import annotations

import unittest

import pandas as pd

from ifind_data_plugin import RequiredMaterialResolver, RetrievalSpec, fetch_data_for_specs


class IsolatingIFindClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def cmd_history_quotation(self, codes, indicators, date):
        del indicators, date
        self.calls.append(list(codes))
        code = codes[0]
        if code.endswith(".LME"):
            raise RuntimeError("THS_HQ failed: errorcode=-4217, errmsg=Permission denied for LME security")
        return pd.DataFrame([{"thscode": code, "settlement": 78000.0}])


class RequiredMaterialScopeTests(unittest.TestCase):
    def test_multi_asset_query_is_not_propagated_to_each_material(self) -> None:
        resolver = RequiredMaterialResolver("domain_dictionary.csv")
        template = {
            "node_template": {
                "nodes": [
                    {
                        "node_id": "S001",
                        "template_description": "行情回顾",
                        "content_guideline": "回顾当前材料。",
                        "required_materials": ["沪铜价格", "宏观利率环境信息"],
                    }
                ]
            }
        }
        specs = resolver.build_specs(
            template,
            "分析沪铜、伦铜、沪铝、伦铝、沪金和COMEX黄金",
            "2025-02-28",
        )
        self.assertEqual(specs[0].codes, ["CU00.SHF"])
        self.assertEqual(specs[1].codes, [])

    def test_explicit_asset_name_remains_a_scoped_fallback(self) -> None:
        resolver = RequiredMaterialResolver("domain_dictionary.csv")
        template = {
            "node_template": {
                "nodes": [
                    {
                        "node_id": "S001",
                        "template_description": "行情回顾",
                        "content_guideline": "回顾行情。",
                        "required_materials": ["板块涨跌幅数据"],
                    }
                ]
            }
        }
        specs = resolver.build_specs(template, "撰写有色金属周报", "2025-02-28", asset_name="沪铜")
        self.assertEqual(specs[0].codes, ["CU00.SHF"])


class PerCodeIsolationTests(unittest.TestCase):
    def test_permission_failure_does_not_discard_successful_code(self) -> None:
        client = IsolatingIFindClient()
        spec = RetrievalSpec(
            node_id="S001",
            state_label="铜行业跟踪",
            required_material="伦铜/沪铜价格",
            codes=["00CAD.LME", "CU00.SHF"],
            indicators=["settlement"],
            date="2025-02-28",
        )
        records, bindings = fetch_data_for_specs([spec], client)
        self.assertEqual(client.calls, [["00CAD.LME"], ["CU00.SHF"]])
        self.assertEqual([record["thscode"] for record in records], ["CU00.SHF"])
        self.assertEqual(bindings[0]["status"], "partial")
        self.assertIn("00CAD.LME", bindings[0]["error"])
        self.assertEqual(
            [(item["code"], item["status"]) for item in bindings[0]["code_results"]],
            [("00CAD.LME", "error"), ("CU00.SHF", "found")],
        )


if __name__ == "__main__":
    unittest.main()
