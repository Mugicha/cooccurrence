#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wordcloud_terms_mecab.py
- SQLite (files/pages) から全テキストを収集
- MeCab で分かち書き & 品詞フィルタ & ストップワード除去
- （任意）原形正規化
- WordCloud 画像を出力（PNG）

依存:
  pip install mecab-python3 wordcloud pillow tqdm
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable, List, Set, Tuple
from collections import Counter

import MeCab
from wordcloud import WordCloud
from tqdm import tqdm

# ========== DB ==========
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

# ========== Stopwords ==========
def load_stopwords(path: Path | None) -> Set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        print(f"[WARN] stopwords not found: {path}")
        return set()
    return {w.strip() for w in p.read_text(encoding="utf-8").splitlines() if w.strip()}

# ========== MeCab ==========
def _guess_pos_from_feature(feature: str) -> str:
    # IPADIC/UniDic どちらでも先頭列が大分類
    if not feature:
        return ""
    return feature.split(",")[0]

def _lemma_from_feature(surface: str, feature: str) -> str:
    """
    できるだけ辞書に合わせて原形（lemma）を返す。
    - IPADIC: 品詞,品詞細1,品詞細2,品詞細3,活用形,活用型,原形,読み,発音
      -> 7列目(インデックス6)が原形（* のこともある）
    - UniDic: pos,pos1,pos2,pos3,cType,cForm,lForm,lemma,orth,pron,...
      -> 8列目(インデックス7)が lemma
    列数で推定し、無ければ surface を返す。
    """
    if not feature:
        return surface
    feats = feature.split(",")
    if len(feats) >= 8:
        # UniDic なら 7:lemma、IPADIC でも 6:原形 が入っていることが多い
        # まず UniDic 位置を試す
        lemma = feats[7]
        if lemma and lemma != "*" and lemma != "":
            return lemma
    if len(feats) >= 7:
        base = feats[6]
        if base and base != "*" and base != "":
            return base
    return surface

def tokenize_mecab(text: str, tagger: MeCab.Tagger,
                   allowed_pos_prefixes: Set[str],
                   stopwords: Set[str],
                   normalize_lemma: bool = False) -> List[str]:
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
        term = _lemma_from_feature(surface, feature) if normalize_lemma else surface
        if term in stopwords:
            continue
        tokens.append(term)
    return tokens

# ========== WordCloud ==========
def build_wordcloud_from_counter(counter: Counter,
                                 font_path: str,
                                 width: int = 1600,
                                 height: int = 1000,
                                 background_color: str = "white",
                                 max_words: int = 300,
                                 prefer_horizontal: float = 0.9) -> WordCloud:
    wc = WordCloud(
        font_path=font_path,
        width=width,
        height=height,
        background_color=background_color,
        max_words=max_words,
        prefer_horizontal=prefer_horizontal,
        collocations=False  # 「東京 大学」を1語にまとめる挙動をOFF
    )
    wc.generate_from_frequencies(counter)
    return wc

# ========== Main ==========
def main():
    ap = argparse.ArgumentParser(description="Build a Japanese word cloud from SQLite (files/pages) with MeCab.")
    ap.add_argument("--db", required=True, type=Path, help="SQLite path")
    ap.add_argument("--name_like", type=str, default=None, help="Filter by files.name LIKE (e.g., %.pdf)")
    ap.add_argument("--only_ext", type=str, default=None, choices=["pdf", "docx"], help="Filter by extension")

    ap.add_argument("--pos", nargs="+", default=["名詞"], help="Allowed POS prefixes (e.g., 名詞 動詞 形容詞)")
    ap.add_argument("--stopwords", type=Path, default=None, help="Stopwords file (one term per line)")
    ap.add_argument("--normalize_lemma", action="store_true", help="Normalize tokens to lemma/base form")

    ap.add_argument("--font_path", type=str, required=True, help="TTF/TTC font path for Japanese text")
    ap.add_argument("--output", type=Path, default=Path("./wordcloud.png"), help="Output PNG path")
    ap.add_argument("--width", type=int, default=1600, help="Image width")
    ap.add_argument("--height", type=int, default=1000, help="Image height")
    ap.add_argument("--background_color", type=str, default="white", help="Background color")
    ap.add_argument("--max_words", type=int, default=300, help="Max words to draw")
    args = ap.parse_args()

    # DB 読み出し
    conn = sqlite3.connect(str(args.db))
    rows = list(fetch_pages(conn, name_like=args.name_like, only_ext=args.only_ext))
    if not rows:
        print("[INFO] No pages selected.")
        return

    tagger = MeCab.Tagger("")  # 必要に応じて "-d <dic_path>"
    allowed_pos = set(args.pos)
    stopwords = load_stopwords(args.stopwords)

    # 語頻度を集計
    counter = Counter()
    for (_fid, _idx, text, _name, _ext) in tqdm(rows, desc="Tokenizing", unit="page"):
        toks = tokenize_mecab(text or "", tagger, allowed_pos, stopwords, normalize_lemma=args.normalize_lemma)
        counter.update(toks)

    if not counter:
        print("[INFO] No tokens found after filtering.")
        return

    # ワードクラウド生成
    wc = build_wordcloud_from_counter(
        counter,
        font_path=args.font_path,
        width=args.width,
        height=args.height,
        background_color=args.background_color,
        max_words=args.max_words
    )

    # 画像保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wc.to_file(str(args.output))
    print(f"[OK] Saved word cloud to: {args.output}")

if __name__ == "__main__":
    main()