#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ingest_docs.py
- 指定フォルダ以下の PDF / DOCX をクロールしてテキスト抽出
- SQLite に ファイル単位/ページ単位 で格納
- フォルダパス(dir)とファイル名(name)を分けて保存

依存:
  pip install pymupdf python-docx tqdm
"""

import argparse
import os
import sys
import sqlite3
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
  path TEXT NOT NULL UNIQUE,   -- 絶対ファイルパス
  dir  TEXT NOT NULL,          -- 絶対ディレクトリパス  ★追加
  name TEXT NOT NULL,          -- ファイル名（拡張子付き）
  ext  TEXT NOT NULL,          -- 拡張子（pdf / docx）
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
    _ensure_dir_column(conn)
    return conn

def _ensure_dir_column(conn: sqlite3.Connection) -> None:
    """
    既存DBに dir 列がなければ追加し、既存レコードも更新する。
    """
    cur = conn.execute("PRAGMA table_info(files);")
    cols = {row[1] for row in cur.fetchall()}  # row[1] は列名
    if "dir" not in cols:
        conn.execute("ALTER TABLE files ADD COLUMN dir TEXT;")
        conn.commit()
        # 既存行に対して dir を埋める（path から導出）
        cur = conn.execute("SELECT id, path FROM files;")
        rows = cur.fetchall()
        for file_id, abspath in rows:
            d = str(Path(abspath).resolve().parent)
            conn.execute("UPDATE files SET dir=? WHERE id=?", (d, file_id))
        conn.commit()

def upsert_file(conn: sqlite3.Connection, path: Path) -> int:
    stat = path.stat()
    mtime_iso = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    ext = path.suffix.lower().lstrip(".")
    dir_path = str(path.resolve().parent)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO files (path, dir, name, ext, size_bytes, mtime)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
          dir=excluded.dir,
          name=excluded.name,
          ext=excluded.ext,
          size_bytes=excluded.size_bytes,
          mtime=excluded.mtime
        """,
        (str(path.resolve()), dir_path, path.name, ext, stat.st_size, mtime_iso),
    )
    conn.commit()
    cur.execute("SELECT id FROM files WHERE path = ?", (str(path.resolve()),))
    return cur.fetchone()[0]

def insert_pages(conn: sqlite3.Connection, file_id: int, pages: List[Tuple[int, str]]) -> None:
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
            txt = page.get_text("text")
            pages.append((i, txt or ""))
    return pages

def _docx_iter_paragraph_texts_with_pagebreaks(doc: Document) -> List[str]:
    from docx.enum.text import WD_BREAK
    pages: List[str] = []
    buff: List[str] = []

    def flush():
        nonlocal buff
        pages.append("\n".join(buff).strip())
        buff = []

    for para in doc.paragraphs:
        if para.text:
            buff.append(para.text)
        for run in para.runs:
            br_elems = run._r.xpath('w:br[@w:type="page"]')
            if br_elems:
                flush() if buff else pages.append("")
        sect_elems = para._p.xpath('w:pPr/w:sectPr')
        if sect_elems:
            flush() if buff else pages.append("")
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
    ap = argparse.ArgumentParser(description="PDF/DOCX テキスト抽出 -> SQLite 格納（dir と name を分離保存）")
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
                continue
            insert_pages(conn, file_id, pages)
            if args.verbose:
                print(f"[OK] {path} -> {len(pages)} pages")
        except Exception as e:
            print(f"[WARN] {path}: {e}", file=sys.stderr)

    print(f"[DONE] {len(paths)} ファイルを処理し、DB に格納しました: {db_path.resolve()}")

if __name__ == "__main__":
    main()