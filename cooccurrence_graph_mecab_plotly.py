#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cooccurrence_graph_mecab_plotly.py

- SQLite (files/pages) からテキスト取得
- MeCab で分かち書き & 品詞フィルタ（IPADIC / UniDic 両対応）
- 共起集計: unit={page|sentence|file} + スライディング窓
- Plotly でインタラクティブ可視化 (HTML)

依存:
  pip install mecab-python3 plotly networkx pandas tqdm
"""

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple, Set
from collections import Counter, defaultdict

import pandas as pd
import networkx as nx
from tqdm import tqdm
import plotly.graph_objects as go

import MeCab

# ------------------------------
# DB 読み出し
# ------------------------------
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

# ------------------------------
# ストップワード
# ------------------------------
def load_stopwords(path: Path) -> Set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        print(f"[WARN] stopwords not found: {path}")
        return set()
    return {w.strip() for w in p.read_text(encoding="utf-8").splitlines() if w.strip()}

# ------------------------------
# 文分割（日本語フレンドリーな簡易版）
# ------------------------------
_SENT_SPLIT_RE = re.compile(
    r"""            # 句点類/感嘆/疑問/ピリオド/改行の繰返しで区切る
    (?<=。|！|？|!|\?)
    |[\r\n]+
    """,
    re.VERBOSE
)

def split_sentences(text: str) -> List[str]:
    # まず改行正規化
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # 分割
    parts = _SENT_SPLIT_RE.split(t)
    # 連結: splitは区切りトークンも含む可能性があるため、空白を綺麗に
    sents = []
    buf = []
    for p in parts:
        if p is None:
            continue
        p = p.strip()
        if not p:
            continue
        sents.append(p)
    return sents

# ------------------------------
# MeCab 形態素解析 & 品詞フィルタ
# ------------------------------
def _guess_pos_from_feature(feature: str) -> str:
    if not feature:
        return ""
    return feature.split(",")[0]  # IPADIC/UniDicとも先頭が大分類

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

# ------------------------------
# 共起カウント
# ------------------------------
def cooccurrence_sequences(seqs: Iterable[List[str]], window: int = 2
                          ) -> Tuple[Counter, Counter]:
    """
    seqs: トークン列のイテレータ（1ページ、1文、1ファイルなど任意の単位）
    window: スライディング窓（右側 j < i+window）
    """
    token_freqs = Counter()
    pair_freqs = Counter()
    for tokens in seqs:
        token_freqs.update(tokens)
        n = len(tokens)
        if n <= 1:
            continue
        for i in range(n):
            jmax = min(n, i + window)
            for j in range(i+1, jmax):
                a, b = tokens[i], tokens[j]
                if a == b:
                    continue
                if a < b:
                    pair = (a, b)
                else:
                    pair = (b, a)
                pair_freqs[pair] += 1
    return token_freqs, pair_freqs

# ------------------------------
# グラフ構築 & 可視化
# ------------------------------
def build_graph(token_freqs: Counter, pair_freqs: Counter, min_freq: int = 1) -> nx.Graph:
    G = nx.Graph()
    for token, freq in token_freqs.items():
        if freq >= min_freq:
            G.add_node(token, freq=freq)
    for (a, b), w in pair_freqs.items():
        if w >= min_freq and a in G and b in G:
            G.add_edge(a, b, weight=w)
    return G

def plot_graph_plotly(G: nx.Graph, out_html: Path, seed: int = 42, k_layout: float = None,
                      title: str = "Co-occurrence Network") -> None:
    if G.number_of_nodes() == 0:
        print("[WARN] Graph is empty; nothing to plot.")
        return

    pos = nx.spring_layout(G, seed=seed, k=k_layout)

    # edges
    edge_x, edge_y = [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1), hoverinfo='none', opacity=0.5
    )

    # nodes
    node_x, node_y, node_text, node_size = [], [], [], []
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        f = d.get("freq", 1)
        node_size.append(max(10, min(60, f * 3)))
        node_text.append(f"{n}<br>freq: {f}")
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[n for n, _ in G.nodes(data=True)],
        textposition='top center',
        marker=dict(size=node_size, line=dict(width=1)),
        hovertext=node_text, hoverinfo='text'
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title, title_x=0.5, showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20), hovermode='closest'
    )
    fig.write_html(str(out_html), include_plotlyjs='cdn', full_html=True)
    print(f"[OK] Saved interactive graph to: {out_html}")

# ------------------------------
# メイン
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Interactive co-occurrence (MeCab + Plotly) from SQLite, unit={page|sentence|file}.")
    ap.add_argument("--db", required=True, type=Path, help="SQLite path (with files/pages)")
    ap.add_argument("--unit", choices=["page", "sentence", "file"], default="page",
                    help="Co-occurrence unit: page (default), sentence, or file")
    ap.add_argument("--name_like", type=str, default=None, help="Filter by files.name LIKE (e.g., %.pdf)")
    ap.add_argument("--only_ext", type=str, default=None, choices=["pdf","docx"], help="Filter by extension")
    ap.add_argument("--pos", nargs="+", default=["名詞"], help="Allowed POS prefixes (e.g., 名詞 動詞 形容詞)")
    ap.add_argument("--window", type=int, default=2, help="Sliding window size in each unit")
    ap.add_argument("--min_freq", type=int, default=2, help="Min freq for nodes/edges to keep")
    ap.add_argument("--stopwords", type=Path, default=None, help="Stopwords file (one term per line)")
    ap.add_argument("--output", type=Path, default=Path("./cooc.html"), help="Output HTML path")
    ap.add_argument("--topn", type=int, default=50, help="Save top-N co-occurrence pairs")
    ap.add_argument("--top_csv", type=Path, default=None, help="If given, write top-N pairs to CSV here")
    ap.add_argument("--layout_k", type=float, default=None, help="spring_layout k (node spacing)")
    ap.add_argument("--seed", type=int, default=42, help="layout random seed")
    args = ap.parse_args()

    conn = sqlite3.connect(str(args.db))
    rows = list(fetch_pages(conn, name_like=args.name_like, only_ext=args.only_ext))
    if not rows:
        print("[INFO] No pages selected.")
        return

    tagger = MeCab.Tagger("")  # 必要なら -d で辞書パス指定
    allowed_pos = set(args.pos)
    stopwords = load_stopwords(args.stopwords)

    # ===== 単位ごとにトークン列を構築 =====
    seqs: List[List[str]] = []

    if args.unit == "page":
        # 各ページを1シーケンスとして扱う
        for (_fid, _idx, text, _name, _ext) in tqdm(rows, desc="Tokenizing (page)", unit="page"):
            toks = tokenize_mecab(text or "", tagger, allowed_pos, stopwords)
            seqs.append(toks)

    elif args.unit == "sentence":
        # 各ページ → 文分割 → 各文を1シーケンス
        for (_fid, _idx, text, _name, _ext) in tqdm(rows, desc="Sentence splitting + tokenizing", unit="page"):
            sents = split_sentences(text or "")
            for s in sents:
                toks = tokenize_mecab(s, tagger, allowed_pos, stopwords)
                if toks:
                    seqs.append(toks)

    elif args.unit == "file":
        # 同一 file_id の全ページを連結して1シーケンス
        by_file: defaultdict[int, List[str]] = defaultdict(list)
        for (fid, _idx, text, _name, _ext) in rows:
            if text:
                by_file[fid].append(text)
        for fid, chunks in tqdm(by_file.items(), desc="Tokenizing (file)", unit="file"):
            toks = tokenize_mecab("\n".join(chunks), tagger, allowed_pos, stopwords)
            seqs.append(toks)

    token_freqs, pair_freqs = cooccurrence_sequences(seqs, window=args.window)

    if args.top_csv:
        df = pd.DataFrame(
            [(a, b, w) for (a, b), w in pair_freqs.most_common(args.topn)],
            columns=["w1", "w2", "weight"]
        )
        df.to_csv(args.top_csv, index=False)
        print(f"[OK] Saved top-{args.topn} pairs to: {args.top_csv}")

    G = build_graph(token_freqs, pair_freqs, min_freq=args.min_freq)
    title = f"Co-occurrence Network (unit={args.unit}, pos={','.join(args.pos)}, window={args.window}, min_freq={args.min_freq})"
    plot_graph_plotly(G, args.output, seed=args.seed, k_layout=args.layout_k, title=title)

if __name__ == "__main__":
    main()
