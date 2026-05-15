"""
Shared logic for loading files into warehouse.duckdb.
Supported: .csv, .xlsx / .xls (all sheets), .db (SQLite, all tables)

Files in inbox/ subfolders load into a matching DuckDB schema:
  inbox/contacts/file.csv  ->  schema: contacts
  inbox/file.csv           ->  schema: main (default)
"""

import re
import time
import duckdb
import pandas as pd
from pathlib import Path

DB    = Path(r"C:\Users\jat27\Documents\Hope Squad\Main\HS Data\warehouse.duckdb")
INBOX = Path(r"C:\Users\jat27\Documents\Hope Squad\Main\HS Data\inbox")
SUPPORTED = {".csv", ".xlsx", ".xls", ".db"}


def to_table_name(*parts: str) -> str:
    """Convert one or more strings into a clean SQL identifier."""
    joined = "_".join(parts)
    name = joined.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def schema_for(path: Path) -> str:
    """Derive the DuckDB schema name from the file's subfolder within inbox/."""
    rel = path.parent.relative_to(INBOX)
    parts = rel.parts
    return to_table_name(parts[0]) if parts else "main"


def connect(retries: int = 5, delay: float = 1.0) -> duckdb.DuckDBPyConnection:
    """Open the warehouse, retrying if another process has it locked."""
    for attempt in range(retries):
        try:
            con = duckdb.connect(str(DB))
            con.execute("INSTALL sqlite; LOAD sqlite")
            return con
        except duckdb.IOException:
            if attempt < retries - 1:
                print(f"  DB locked, retrying in {delay}s…")
                time.sleep(delay)
            else:
                raise


def _ensure_schema(con: duckdb.DuckDBPyConnection, schema: str):
    if schema != "main":
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')


def load_file(path: Path) -> list[str]:
    """
    Load a file into the warehouse as one or more tables.
    Returns the list of qualified table names (schema.table) that were created/replaced.
    """
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        return []

    # Brief pause so the OS finishes writing the file
    time.sleep(0.5)

    schema = schema_for(path)
    created = []
    con = connect()

    try:
        _ensure_schema(con, schema)

        if ext == ".csv":
            tbl = to_table_name(path.stem)
            con.execute(
                f'CREATE OR REPLACE TABLE "{schema}"."{tbl}" AS '
                f"SELECT * FROM read_csv_auto('{path.as_posix()}', quote='\"')"
            )
            created.append(f"{schema}.{tbl}")

        elif ext in (".xlsx", ".xls"):
            xl = pd.ExcelFile(path)
            for sheet in xl.sheet_names:
                tbl = (
                    to_table_name(path.stem)
                    if len(xl.sheet_names) == 1
                    else to_table_name(path.stem, sheet)
                )
                df = xl.parse(sheet)
                con.register("_tmp", df)
                con.execute(f'CREATE OR REPLACE TABLE "{schema}"."{tbl}" AS SELECT * FROM _tmp')
                created.append(f"{schema}.{tbl}")

        elif ext == ".db":
            safe_alias = to_table_name(path.stem) + "_src"
            con.execute(
                f"ATTACH '{path.as_posix()}' AS {safe_alias} (TYPE sqlite, READ_ONLY)"
            )
            sqlite_tables = con.execute(
                f"SELECT table_name FROM information_schema.tables "
                f"WHERE table_schema = '{safe_alias}'"
            ).fetchall()
            for (sqlite_tbl,) in sqlite_tables:
                tbl = to_table_name(path.stem, sqlite_tbl)
                con.execute(
                    f'CREATE OR REPLACE TABLE "{schema}"."{tbl}" AS '
                    f"SELECT * FROM {safe_alias}.{sqlite_tbl}"
                )
                created.append(f"{schema}.{tbl}")
            con.execute(f"DETACH {safe_alias}")

    finally:
        con.close()

    return created


def drop_file(path: Path) -> list[str]:
    """
    Drop all tables that were loaded from the given file.
    Returns the list of qualified table names that were dropped.
    """
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        return []

    schema = schema_for(path)
    stem = to_table_name(path.stem)
    con = connect()

    rows = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = ?",
        [schema]
    ).fetchall()
    all_tables = [row[0] for row in rows]
    to_drop = [t for t in all_tables if t == stem or t.startswith(stem + "_")]

    dropped = []
    for tbl in to_drop:
        con.execute(f'DROP TABLE IF EXISTS "{schema}"."{tbl}"')
        dropped.append(f"{schema}.{tbl}")

    con.close()
    return dropped
