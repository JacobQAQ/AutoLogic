# -*- coding: utf-8 -*-
"""
iFinD HTTP API data retrieval tool for LogicRAG.

Functions:
1. Futures query:
   user futures name -> domain_dictionary.csv -> CODE -> iFinD historical quotation API

2. Stock query:
   user stock name -> get_thscode -> CODE -> iFinD historical quotation API

CSV format expected:
    CODE,Name
    AU00.SHF,沪金连续
    CU00.SHF,沪铜连续
    @GC0Y.CMX,纽约金连一

Run examples:
    python ifind_api_tool.py --name 沪金连续 --date 2025-02-28 --dict domain_dictionary.csv
    python ifind_api_tool.py --name 同花顺 --date 2025-02-28 --asset-type auto
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
import requests


BASE_URL = "https://quantapi.51ifind.com/api/v1"


# =========================
# Indicator Mapping
# =========================

# Your Appendix-B-style indicator names -> iFinD HTTP historical quotation indicators.
APPENDIX_B_TO_HTTP_HISTORY = {
    "ths_pre_settle_future": "preSettlement",
    "ths_settle_future": "settlement",
    "ths_settle_chg_future": "change_settlement",
    "ths_chg_future": "change",
    "ths_chg_ratio_future": "changeRatio",
    "ths_settle_chg_ratio_future": "chg_settlement",
    "ths_vol_future": "volume",
    "ths_open_interest_future": "openInterest",
}

DEFAULT_FUTURES_INDICATORS = [
    "ths_pre_settle_future",
    "ths_settle_future",
    "ths_settle_chg_future",
    "ths_chg_future",
    "ths_chg_ratio_future",
    "ths_settle_chg_ratio_future",
    "ths_vol_future",
    "ths_open_interest_future",
]

DEFAULT_STOCK_INDICATORS = [
    "open",
    "high",
    "low",
    "close",
    "change",
    "changeRatio",
    "volume",
]


# =========================
# Query Keys
# =========================

@dataclass
class QueryKeys:
    """
    Three-dimensional query key:
        CODES, INDICATORS, DATE
    """
    codes: Sequence[str]
    indicators: Sequence[str]
    date: str
    startdate: Optional[str] = None
    enddate: Optional[str] = None
    interval: str = "D"
    fill: str = "Blank"


# =========================
# Futures Dictionary
# =========================

class FuturesDomainDictionary:
    """
    Resolve futures names into iFinD CODE using domain_dictionary.csv.

    Required CSV columns:
        CODE, Name
    """

    def __init__(self, csv_path: str = "domain_dictionary.csv") -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Cannot find domain dictionary: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path, encoding="utf-8-sig")

        required_cols = {"CODE", "Name"}
        if not required_cols.issubset(set(self.df.columns)):
            raise ValueError(
                f"domain_dictionary.csv must contain columns {required_cols}, "
                f"but got columns: {self.df.columns.tolist()}"
            )

        self.df["CODE"] = self.df["CODE"].astype(str).str.strip()
        self.df["Name"] = self.df["Name"].astype(str).str.strip()
        self.df["_norm_name"] = self.df["Name"].apply(self._normalize)

    @staticmethod
    def _normalize(text: str) -> str:
        text = str(text).strip().lower()
        text = re.sub(r"\s+", "", text)
        text = text.replace("（", "(").replace("）", ")")
        text = text.replace("，", ",")
        return text

    @staticmethod
    def _remove_query_words(text: str) -> str:
        words = [
            "查询", "获取", "提取", "价格", "行情", "数据", "期货",
            "合约", "收盘价", "结算价", "涨跌幅", "成交量", "持仓量",
            "走势", "日报", "周报", "月报",
        ]
        for w in words:
            text = text.replace(w, "")
        return text

    def resolve(self, query_name: str) -> str:
        """
        Resolve futures name to CODE.

        Matching priority:
        1. Exact match:
           沪金连续 -> AU00.SHF
        2. Query contains dictionary name:
           查询沪金连续价格 -> AU00.SHF
        3. Dictionary name contains query:
           沪金 -> prefer 沪金连续 / 沪金主连
        4. Stop-word fallback:
           查询沪金价格 -> 沪金
        """
        q = self._normalize(query_name)

        # 1. Exact match
        exact = self.df[self.df["_norm_name"] == q]
        if not exact.empty:
            return str(exact.iloc[0]["CODE"])

        # 2. Query contains dictionary name
        contains_name = self.df[self.df["_norm_name"].apply(lambda x: bool(x) and x in q)]
        if not contains_name.empty:
            contains_name = contains_name.assign(_len=contains_name["_norm_name"].str.len())
            contains_name = contains_name.sort_values("_len", ascending=False)
            return str(contains_name.iloc[0]["CODE"])

        # 3. Dictionary name contains query
        name_contains = self.df[self.df["_norm_name"].apply(lambda x: bool(q) and q in x)]
        if not name_contains.empty:
            ranked = self._rank_candidates(name_contains)
            return str(ranked.iloc[0]["CODE"])

        # 4. Stop-word fallback
        q2 = self._normalize(self._remove_query_words(query_name))
        fallback = self.df[self.df["_norm_name"].apply(lambda x: bool(q2) and q2 in x)]
        if not fallback.empty:
            ranked = self._rank_candidates(fallback)
            return str(ranked.iloc[0]["CODE"])

        raise KeyError(f"Cannot resolve futures code from query: {query_name}")

    def _rank_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        Prefer continuous/main contracts when the user only inputs a general name.
        """

        def score(name: str) -> int:
            name = str(name)
            if "连续" in name:
                return 100
            if "主连" in name:
                return 95
            if "连一" in name:
                return 85
            if "加权" in name:
                return 70
            if re.search(r"\d{3,4}", name):
                return 50
            return 10

        df = candidates.copy()
        df["_rank"] = df["Name"].apply(score)
        return df.sort_values("_rank", ascending=False)

    def show_candidates(self, query_name: str, topk: int = 10) -> pd.DataFrame:
        """
        Show possible matches for debugging.
        """
        q = self._normalize(query_name)
        q2 = self._normalize(self._remove_query_words(query_name))

        matched = self.df[
            self.df["_norm_name"].apply(
                lambda x: (q and (q in x or x in q)) or (q2 and q2 in x)
            )
        ]

        if matched.empty:
            return pd.DataFrame(columns=["CODE", "Name"])

        return self._rank_candidates(matched)[["CODE", "Name"]].head(topk)


# =========================
# iFinD HTTP Client
# =========================

class IFINDHttpClient:
    def __init__(
        self,
        refresh_token: Optional[str] = None,
        access_token: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self.refresh_token = refresh_token or os.getenv("IFIND_REFRESH_TOKEN")
        self.access_token = access_token or os.getenv("IFIND_ACCESS_TOKEN")
        self.timeout = timeout
        self.session = requests.Session()

    def get_access_token(self, force_update: bool = False) -> str:
        """
        Get access_token from refresh_token.

        force_update=False:
            use get_access_token endpoint.

        force_update=True:
            use update_access_token endpoint.
            Warning: this invalidates old access_tokens.
        """
        if self.access_token and not force_update:
            return self.access_token

        if not self.refresh_token:
            raise ValueError(
                "Missing refresh_token. Set environment variable IFIND_REFRESH_TOKEN "
                "or pass --refresh-token when running this script."
            )

        endpoint = "update_access_token" if force_update else "get_access_token"
        url = f"{BASE_URL}/{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "refresh_token": self.refresh_token,
        }

        response = self.session.post(url=url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        payload = self._safe_json(response)

        try:
            token = payload["data"]["access_token"]
        except KeyError as exc:
            raise RuntimeError(f"Failed to parse access_token from response: {payload}") from exc

        self.access_token = token
        return token

    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST to iFinD HTTP API with access_token.
        """
        token = self.get_access_token()
        url = f"{BASE_URL}/{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "access_token": token,
            "Accept-Encoding": "gzip,deflate",
        }

        response = self.session.post(
            url=url,
            json=data,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = self._safe_json(response)

        error_code = payload.get("errorcode")
        if error_code not in (None, 0, "0"):
            raise RuntimeError(
                f"iFinD API returned errorcode={error_code}, "
                f"errmsg={payload.get('errmsg')}, payload={payload}"
            )

        return payload

    def query_history(self, query: QueryKeys) -> pd.DataFrame:
        """
        Query historical quotation data.

        Endpoint:
            /api/v1/cmd_history_quotation

        Required:
            codes, indicators, startdate, enddate
        """
        startdate = query.startdate or query.date
        enddate = query.enddate or query.date

        indicators = [
            APPENDIX_B_TO_HTTP_HISTORY.get(x, x)
            for x in query.indicators
        ]

        request_body = {
            "codes": ",".join(query.codes),
            "indicators": ",".join(indicators),
            "startdate": startdate,
            "enddate": enddate,
            "functionpara": {
                "Interval": query.interval,
                "Fill": query.fill,
            },
        }

        raw = self.post("cmd_history_quotation", request_body)
        df = parse_ifind_response(raw)

        df.insert(0, "query_date", query.date)
        df.insert(1, "query_codes", ",".join(query.codes))
        df.insert(2, "query_indicators", ",".join(indicators))
        return df

    def resolve_thscode(self, keyword: str, mode: str = "secname") -> pd.DataFrame:
        """
        Resolve stock name or raw stock code into iFinD thscode.

        mode:
            secname: keyword is stock name, e.g. 同花顺
            seccode: keyword is raw code, e.g. 300033
        """
        if mode not in {"secname", "seccode"}:
            raise ValueError("mode must be either 'secname' or 'seccode'.")

        body = {
            mode: keyword,
            "mode": mode,
        }

        raw = self.post("get_thscode", body)
        return parse_ifind_response(raw)

    def resolve_stock_code(self, secname: str) -> str:
        """
        Resolve stock name to thscode.
        Example:
            同花顺 -> 300033.SZ
        """
        df = self.resolve_thscode(secname, mode="secname")

        possible_cols = [
            "thscode",
            "thsCode",
            "THSCODE",
            "code",
            "CODE",
            "CODES",
            "证券代码",
        ]

        code_col = next((c for c in possible_cols if c in df.columns), None)

        if code_col is None:
            raise RuntimeError(
                f"Cannot find thscode column from get_thscode response. "
                f"Columns: {df.columns.tolist()}; data: {df.head().to_dict(orient='records')}"
            )

        return str(df.iloc[0][code_col]).strip()

    def query_price_by_name(
        self,
        secname: str,
        date: str,
        indicators: Union[str, Sequence[str]] = ",".join(DEFAULT_STOCK_INDICATORS),
        output_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Query stock price by stock name.

        Example:
            同花顺 -> get_thscode -> 300033.SZ -> query_history
        """
        code = self.resolve_stock_code(secname)

        if isinstance(indicators, str):
            indicator_list = [x.strip() for x in indicators.split(",") if x.strip()]
        else:
            indicator_list = list(indicators)

        query = QueryKeys(
            codes=[code],
            indicators=indicator_list,
            date=date,
            startdate=date,
            enddate=date,
            interval="D",
            fill="Blank",
        )

        df = self.query_history(query)
        df.insert(0, "asset_type", "stock")
        df.insert(1, "resolved_name", secname)
        df.insert(2, "resolved_code", code)

        if output_csv:
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")

        return df

    def query_futures_by_dictionary(
        self,
        query_name: str,
        date: str,
        dictionary_path: str = "domain_dictionary.csv",
        indicators: Optional[Union[str, Sequence[str]]] = None,
        output_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Query futures data using domain_dictionary.csv.

        Example:
            沪金连续 -> AU00.SHF
            纽约金主连 -> @GC0W.CMX
        """
        dictionary = FuturesDomainDictionary(dictionary_path)
        code = dictionary.resolve(query_name)

        if indicators is None:
            indicator_list = DEFAULT_FUTURES_INDICATORS
        elif isinstance(indicators, str):
            indicator_list = [x.strip() for x in indicators.split(",") if x.strip()]
        else:
            indicator_list = list(indicators)

        query = QueryKeys(
            codes=[code],
            indicators=indicator_list,
            date=date,
            startdate=date,
            enddate=date,
            interval="D",
            fill="Blank",
        )

        df = self.query_history(query)
        df.insert(0, "asset_type", "future")
        df.insert(1, "resolved_name", query_name)
        df.insert(2, "resolved_code", code)

        if output_csv:
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")

        return df

    def query_asset_by_name(
        self,
        query_name: str,
        date: str,
        asset_type: str = "auto",
        dictionary_path: str = "domain_dictionary.csv",
        indicators: Optional[Union[str, Sequence[str]]] = None,
        output_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Unified query entrance.

        asset_type:
            auto    : try futures dictionary first, then fallback to stock
            future  : force futures dictionary
            futures : force futures dictionary
            stock   : force stock query by get_thscode
        """
        if asset_type not in {"auto", "future", "futures", "stock"}:
            raise ValueError("asset_type must be one of {'auto', 'future', 'futures', 'stock'}.")

        if asset_type in {"future", "futures"}:
            return self.query_futures_by_dictionary(
                query_name=query_name,
                date=date,
                dictionary_path=dictionary_path,
                indicators=indicators,
                output_csv=output_csv,
            )

        if asset_type == "stock":
            stock_indicators = indicators or DEFAULT_STOCK_INDICATORS
            return self.query_price_by_name(
                secname=query_name,
                date=date,
                indicators=stock_indicators,
                output_csv=output_csv,
            )

        # auto mode: futures first, stock fallback.
        try:
            return self.query_futures_by_dictionary(
                query_name=query_name,
                date=date,
                dictionary_path=dictionary_path,
                indicators=indicators,
                output_csv=output_csv,
            )
        except (KeyError, FileNotFoundError, ValueError):
            stock_indicators = indicators or DEFAULT_STOCK_INDICATORS
            return self.query_price_by_name(
                secname=query_name,
                date=date,
                indicators=stock_indicators,
                output_csv=output_csv,
            )

    @staticmethod
    def _safe_json(response: requests.Response) -> Dict[str, Any]:
        try:
            return response.json()
        except Exception:
            text = response.content.decode("utf-8", errors="replace")
            return json.loads(text)


# =========================
# Response Parser
# =========================

def parse_ifind_response(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Robust parser for common iFinD HTTP API response structures.

    Common top-level fields:
        errorcode, errmsg, tables, datatype, inputParams, perf, dataVol
    """
    if not payload:
        return pd.DataFrame()

    tables = payload.get("tables")

    if tables is None and isinstance(payload.get("data"), dict):
        tables = payload["data"].get("tables") or payload["data"]

    if tables is None:
        return pd.DataFrame([flatten_dict(payload)])

    frames: List[pd.DataFrame] = []

    if isinstance(tables, list):
        for item in tables:
            frames.append(parse_single_table(item))
    elif isinstance(tables, dict):
        if "table" in tables or "thscode" in tables or "code" in tables:
            frames.append(parse_single_table(tables))
        else:
            for code, content in tables.items():
                frames.append(parse_single_table({"thscode": code, "table": content}))
    else:
        frames.append(pd.DataFrame([{"raw_tables": tables}]))

    frames = [x for x in frames if x is not None and not x.empty]

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def parse_single_table(item: Any) -> pd.DataFrame:
    if not isinstance(item, dict):
        return pd.DataFrame([{"value": item}])

    thscode = (
        item.get("thscode")
        or item.get("thsCode")
        or item.get("code")
        or item.get("CODE")
        or item.get("CODES")
    )

    table = item.get("table") or item.get("data") or item

    if isinstance(table, list):
        df = pd.DataFrame(table)

    elif isinstance(table, dict):
        # dict of lists -> table
        if any(isinstance(v, list) for v in table.values()):
            df = pd.DataFrame(table)
        else:
            df = pd.DataFrame([flatten_dict(table)])

    else:
        df = pd.DataFrame([{"value": table}])

    if thscode is not None and "thscode" not in df.columns:
        df.insert(0, "thscode", thscode)

    return df


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dict for diagnostic output.
    """
    items: Dict[str, Any] = {}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)

        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            items[new_key] = json.dumps(v, ensure_ascii=False)
        else:
            items[new_key] = v

    return items


# =========================
# CLI
# =========================

def parse_indicator_arg(indicators: Optional[str]) -> Optional[List[str]]:
    if not indicators:
        return None
    return [x.strip() for x in indicators.split(",") if x.strip()]


def safe_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[\\/:*?\"<>|,\s]+", "_", text)
    return text.strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="iFinD HTTP API tool for LogicRAG.")
    parser.add_argument("--name", type=str, default=None, help="Asset name, e.g., 沪金连续, 同花顺")
    parser.add_argument("--date", type=str, default="2025-02-28", help="Query date, e.g., 2025-02-28")
    parser.add_argument("--asset-type", type=str, default="auto", choices=["auto", "stock", "future", "futures"])
    parser.add_argument("--dict", type=str, default="domain_dictionary.csv", help="Path to domain_dictionary.csv")
    parser.add_argument("--indicators", type=str, default=None, help="Comma-separated indicators")
    parser.add_argument("--codes", type=str, default=None, help="Direct iFinD codes, comma-separated")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    parser.add_argument("--refresh-token", type=str, default=None, help="iFinD refresh_token")
    parser.add_argument("--access-token", type=str, default=None, help="iFinD access_token")
    parser.add_argument("--force-update-token", action="store_true", help="Force update access_token")
    parser.add_argument("--show-candidates", action="store_true", help="Show futures dictionary candidates only")

    args = parser.parse_args()

    if args.show_candidates:
        if not args.name:
            raise ValueError("--show-candidates requires --name")
        dictionary = FuturesDomainDictionary(args.dict)
        print(dictionary.show_candidates(args.name).to_string(index=False))
        return

    client = IFINDHttpClient(
        refresh_token=args.refresh_token,
        access_token=args.access_token,
    )

    client.get_access_token(force_update=args.force_update_token)

    indicators = parse_indicator_arg(args.indicators)

    if args.codes:
        code_list = [x.strip() for x in args.codes.split(",") if x.strip()]
        if indicators is None:
            indicators = DEFAULT_FUTURES_INDICATORS if args.asset_type in {"future", "futures"} else DEFAULT_STOCK_INDICATORS

        query = QueryKeys(
            codes=code_list,
            indicators=indicators,
            date=args.date,
            startdate=args.date,
            enddate=args.date,
            interval="D",
            fill="Blank",
        )
        df = client.query_history(query)

        output_csv = args.out or f"ifind_codes_{safe_filename(args.date)}.csv"
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")

        print(f"[OK] Retrieved {len(df)} rows.")
        print(f"[OK] Saved to: {output_csv}")
        print(df.head(20).to_string(index=False))
        return

    if not args.name:
        raise ValueError("Please provide --name or --codes.")

    output_csv = args.out or f"ifind_{safe_filename(args.name)}_{safe_filename(args.date)}.csv"

    df = client.query_asset_by_name(
        query_name=args.name,
        date=args.date,
        asset_type=args.asset_type,
        dictionary_path=args.dict,
        indicators=indicators,
        output_csv=output_csv,
    )

    print(f"[OK] Retrieved {len(df)} rows.")
    print(f"[OK] Saved to: {output_csv}")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()