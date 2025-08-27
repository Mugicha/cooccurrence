#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pareto_terms_mecab_plotly.py
- SQLite (files/pages) のテキストを集計し、語の出現頻度でパレート図を出力
- MeCab 形態素解析 + 品詞フィルタ + ストップワード対応
- 出力: Plotly の HTML（棒: 度数 / 折れ線: 累積比率）

依存:
  pip install mecab-python3 plotly pandas tqdm
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable, List, Set, Tuple
from collections import Counter

import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
import MeCab

# =========================
# DB 読み出し
# =========================

SQL_BASE = """
SELECT p.file_id, p.page_index, p.text, f.name, f.ext
FROM pages p
JOIN files f ON f.id = p.file_id
WHERE 1=1
"""

def fetch_pages(conn: sqlite3.Connection, name_like: str = None, only_ext: str = None
               ) -> Iterable[Tuple[int,int,str,str,str]]:
    query = SQL_BASE
    params = []
    if name_like:
        query += " AND f.name LIKE ?"
        params.append(name_like)
    if only_ext:
        query += " AND f.ext = ?"
        params.append(only_ext.lower())
    query += " ORDER BY p.file_id, p.page_index"
    cur = conn.execute(query, params)
    yield from cur  # (file_id, page_index, text, name, ext)

# =========================
# ストップワード
# =========================

def load_stopwords(path: Path | None) -> Set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        print(f"[WARN] stopwords not found: {path}")
        return set()
    return {w.strip() for w in p.read_text(encoding="utf-8").splitlines() if w.strip()}

# =========================
# MeCab 形態素解析
# =========================

def _guess_pos_from_feature(feature: str) -> str:
    # IPADIC/UniDic いずれも先頭列が品詞大分類
    if not feature:
        return ""
    return feature.split(",")[0]

def tokenize_mecab(text: str, tagger: MeCab.Tagger,
                   allowed_pos_prefixes: Set[str],
                   stopwords: Set[str]) -> List[str]:
    tokens: List[str] = []
    node_str = tagger.parse(text)
    if not node_str:
        return tokens
    for line in node_str.splitlines():
        if line == "EOS" or not line.strip():
            continue
        if "\t" not in line:
            continue
        surface, feature = line.split("\t", 1)
        surface = surface.strip()
        if not surface:
            continue
        pos = _guess_pos_from_feature(feature)
        if allowed_pos_prefixes and not any(pos.startswith(pfx) for pfx in allowed_pos_prefixes):
            continue
        if surface in stopwords:
            continue
        tokens.append(surface)
    return tokens

# =========================
# パレート図の作成
# =========================

def make_pareto_figure(df_top: pd.DataFrame, title: str) -> go.Figure:
    """
    df_top: columns = ['term','count','ratio','cum_ratio','rank']
    """
    fig = go.Figure()

    # 棒（度数）
    fig.add_trace(go.Bar(
        x=df_top["term"],
        y=df_top["count"],
        name="頻度",
        yaxis="y1",
        hovertemplate="語: %{x}<br>頻度: %{y}<extra></extra>"
    ))

    # 折れ線（累積比率 %）
    fig.add_trace(go.Scatter(
        x=df_top["term"],
        y=(df_top["cum_ratio"] * 100.0),
        name="累積比率",
        yaxis="y2",
        mode="lines+markers",
        hovertemplate="語: %{x}<br>累積比率: %{y:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        title=title, title_x=0.5,
        xaxis=dict(title="語（出現頻度順）"),
        yaxis=dict(title="頻度", rangemode="tozero"),
        yaxis2=dict(
            title="累積比率（%）",
            overlaying="y",
            side="right",
            rangemode="tozero",
            tickformat=".0f"
        ),
        bargap=0.2,
        hovermode="x unified",
        margin=dict(l=40, r=60, t=60, b=80),
        showlegend=True,
    )
    return fig

# =========================
# メイン
# =========================

def main():
    ap = argparse.ArgumentParser(description="Make a Pareto chart (term frequency) from SQLite (files/pages).")
    ap.add_argument("--db", required=True, type=Path, help="SQLite path")
    ap.add_argument("--name_like", type=str, default=None, help="Filter by files.name LIKE (e.g., %.pdf)")
    ap.add_argument("--only_ext", type=str, default=None, choices=["pdf", "docx"], help="Filter by extension")
    ap.add_argument("--pos", nargs="+", default=["名詞"], help="Allowed POS prefixes (e.g., 名詞 動詞 形容詞)")
    ap.add_argument("--stopwords", type=Path, default=None, help="Stopwords file (one term per line)")
    ap.add_argument("--topk", type=int, default=30, help="Top-K terms to display in Pareto chart")
    ap.add_argument("--output", type=Path, default=Path("./pareto_terms.html"), help="Output HTML path for the chart")
    ap.add_argument("--freq_csv", type=Path, default=None, help="If given, save all term frequency to CSV")
    args = ap.parse_args()

    conn = sqlite3.connect(str(args.db))
    rows = list(fetch_pages(conn, name_like=args.name_like, only_ext=args.only_ext))
    if not rows:
        print("[INFO] No pages selected.")
        return

    tagger = MeCab.Tagger("")  # 必要に応じて辞書パス指定（例: "-d /opt/homebrew/lib/mecab/dic/ipadic"）
    allowed_pos = set(args.pos)
    stopwords = load_stopwords(args.stopwords)

    # 全テキストを横断して語頻度を集計
    counter = Counter()
    for (_fid, _idx, text, _name, _ext) in tqdm(rows, desc="Tokenizing", unit="page"):
        toks = tokenize_mecab(text or "", tagger, allowed_pos, stopwords)
        counter.update(toks)

    if not counter:
        print("[INFO] No tokens found after filtering.")
        return

    # DataFrame 化 & ランク付け & 比率・累積比率
    df = pd.DataFrame(counter.items(), columns=["term", "count"]).sort_values("count", ascending=False)
    total = df["count"].sum()
    df["ratio"] = df["count"] / total
    df["cum_ratio"] = df["ratio"].cumsum()
    df["rank"] = range(1, len(df) + 1)

    # CSV（全文）を書き出す場合
    if args.freq_csv:
        df.to_csv(args.freq_csv, index=False)
        print(f"[OK] Saved frequency list to: {args.freq_csv}")

    # 上位 K をパレート図に
    df_top = df.head(args.topk).copy()
    fig = make_pareto_figure(
        df_top,
        title=f"Pareto Chart of Terms (pos={','.join(args.pos)}, top={args.topk})"
    )
    fig.write_html(str(args.output), include_plotlyjs="cdn", full_html=True)
    print(f"[OK] Saved Pareto chart to: {args.output}")

if __name__ == "__main__":
    main()