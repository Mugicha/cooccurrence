#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ingest_docs.py
- 指定フォルダ以下の PDF / DOCX をクロールしてテキスト抽出
- SQLite に ファイル単位/ページ単位 で格納

依存:
  pip install pymupdf python-docx tqdm
"""

import argparse
import os
import sys
import sqlite3
import time
from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Tuple

import fitz  # PyMuPDF
from docx import Document
from tqdm import tqdm


# =========================
# DB ユーティリティ
# =========================

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  ext  TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  mtime TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  file_id INTEGER NOT NULL,
  page_index INTEGER NOT NULL,
  text TEXT,
  FOREIGN KEY(file_id) REFERENCES files(id)
);

CREATE INDEX IF NOT EXISTS idx_pages_file_id ON pages(file_id);
CREATE INDEX IF NOT EXISTS idx_pages_file_id_page ON pages(file_id, page_index);
"""

def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(SCHEMA_SQL)
    return conn

def upsert_file(conn: sqlite3.Connection, path: Path) -> int:
    stat = path.stat()
    mtime_iso = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    ext = path.suffix.lower().lstrip(".")
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO files (path, name, ext, size_bytes, mtime)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
          name=excluded.name,
          ext=excluded.ext,
          size_bytes=excluded.size_bytes,
          mtime=excluded.mtime
        """,
        (str(path.resolve()), path.name, ext, stat.st_size, mtime_iso),
    )
    conn.commit()
    # 既存/新規に関わらず id を返す
    cur.execute("SELECT id FROM files WHERE path = ?", (str(path.resolve()),))
    return cur.fetchone()[0]

def insert_pages(conn: sqlite3.Connection, file_id: int, pages: List[Tuple[int, str]]) -> None:
    # 既存ページを消して入れ直す（更新容易のため）
    conn.execute("DELETE FROM pages WHERE file_id = ?", (file_id,))
    conn.executemany(
        "INSERT INTO pages (file_id, page_index, text) VALUES (?, ?, ?)",
        [(file_id, idx, txt) for idx, txt in pages]
    )
    conn.commit()


# =========================
# 抽出ロジック
# =========================

def extract_pdf_pages(path: Path) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc):
            # プレーンテキスト。必要に応じて "blocks" や "rawdict" も検討可
            txt = page.get_text("text")
            pages.append((i, txt or ""))
    return pages

def _docx_iter_paragraph_texts_with_pagebreaks(doc: Document) -> List[str]:
    """
    DOCXはレイアウトエンジンがないため厳密なページ概念がありません。
    ここでは「改ページ（WD_BREAK.PAGE）」と「セクション区切り（sectPr）」をページ境界とみなして分割します。
    ページ区切りがない文書は単一ページとして扱います。
    """
    from docx.enum.text import WD_BREAK

    pages: List[str] = []
    buff: List[str] = []

    def flush():
        nonlocal buff
        pages.append("\n".join(buff).strip())
        buff = []

    for para in doc.paragraphs:
        # 段落テキスト
        if para.text:
            buff.append(para.text)

        # ラン中の改ページを検出
        for run in para.runs:
            # w:br[@w:type="page"]
            br_elems = run._r.xpath('w:br[@w:type="page"]')
            if br_elems:
                if buff:
                    flush()
                else:
                    pages.append("")  # 連続ページ区切り対策

        # セクション区切り（多くは新ページ開始）
        sect_elems = para._p.xpath('w:pPr/w:sectPr')
        if sect_elems:
            if buff:
                flush()
            else:
                pages.append("")

    if buff or not pages:
        flush()

    return pages

def extract_docx_pages(path: Path) -> List[Tuple[int, str]]:
    doc = Document(str(path))
    texts = _docx_iter_paragraph_texts_with_pagebreaks(doc)
    return [(i, t) for i, t in enumerate(texts)]

def is_target_file(path: Path) -> bool:
    return path.suffix.lower() in {".pdf", ".docx"}

def walk_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and is_target_file(p):
            yield p


# =========================
# メイン
# =========================

def main():
    ap = argparse.ArgumentParser(description="PDF/DOCX テキスト抽出 -> SQLite 格納")
    ap.add_argument("--root", required=True, type=Path, help="入力フォルダ（再帰）")
    ap.add_argument("--db", required=True, type=Path, help="出力SQLiteファイルパス")
    ap.add_argument("--verbose", action="store_true", help="詳細ログ")
    args = ap.parse_args()

    root: Path = args.root
    db_path: Path = args.db

    if not root.exists():
        print(f"[ERROR] root not found: {root}", file=sys.stderr)
        sys.exit(1)

    conn = open_db(db_path)

    paths = list(walk_files(root))
    if not paths:
        print("[INFO] 対象ファイルが見つかりませんでした。拡張子: .pdf, .docx")
        return

    for path in tqdm(paths, desc="Indexing", unit="file"):
        try:
            file_id = upsert_file(conn, path)
            ext = path.suffix.lower()
            if ext == ".pdf":
                pages = extract_pdf_pages(path)
            elif ext == ".docx":
                pages = extract_docx_pages(path)
            else:
                continue  # 不達
            insert_pages(conn, file_id, pages)

            if args.verbose:
                print(f"[OK] {path} -> {len(pages)} pages")
        except Exception as e:
            print(f"[WARN] {path}: {e}", file=sys.stderr)
            # 続行

    print(f"[DONE] {len(paths)} ファイルを処理し、DB に格納しました: {db_path.resolve()}")

if __name__ == "__main__":
    main()
