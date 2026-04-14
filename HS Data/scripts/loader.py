"""
Shared logic for loading files into warehouse.duckdb.
Supported: .csv, .xlsx / .xls (all sheets), .db (SQLite, all tables)
"""

import re
import time
import duckdb
import pandas as pd
from pathlib import Path

DB   = Path(r"C:\Users\jat27\Documents\Hope Squad\Main\HS Data\warehouse.duckdb")
INBOX = Path(r"C:\Users\jat27\Documents\Hope Squad\Main\HS Data\inbox")
SUPPORTED = {".csv", ".xlsx", ".xls", ".db"}


def to_table_name(*parts: str) -> str:
    """Convert one or more strings into a clean SQL table name."""
    joined = "_".join(parts)
    name = joined.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


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


def load_file(path: Path) -> list[str]:
    """
    Load a file into the warehouse as one or more tables.
    Returns the list of table names that were created/replaced.
    """
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        return []

    # Brief pause so the OS finishes writing the file
    time.sleep(0.5)

    created = []
    con = connect()

    try:
        if ext == ".csv":
            tbl = to_table_name(path.stem)
            con.execute(
                f"CREATE OR REPLACE TABLE {tbl} AS "
                f"SELECT * FROM read_csv_auto('{path.as_posix()}', quote='\"')"
            )
            created.append(tbl)

        elif ext in (".xlsx", ".xls"):
            xl = pd.ExcelFile(path)
            for sheet in xl.sheet_names:
                # Single-sheet file → use filename only; multi-sheet → filename_sheet
                tbl = (
                    to_table_name(path.stem)
                    if len(xl.sheet_names) == 1
                    else to_table_name(path.stem, sheet)
                )
                df = xl.parse(sheet)
                con.register("_tmp", df)
                con.execute(f"CREATE OR REPLACE TABLE {tbl} AS SELECT * FROM _tmp")
                created.append(tbl)

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
                    f"CREATE OR REPLACE TABLE {tbl} AS "
                    f"SELECT * FROM {safe_alias}.{sqlite_tbl}"
                )
                created.append(tbl)
            con.execute(f"DETACH {safe_alias}")

    finally:
        con.close()

    return created


def drop_file(path: Path) -> list[str]:
    """
    Drop all tables that were loaded from the given file.
    Returns the list of table names that were dropped.
    """
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        return []

    stem = to_table_name(path.stem)
    con = connect()
    all_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    to_drop = [t for t in all_tables if t == stem or t.startswith(stem + "_")]

    for tbl in to_drop:
        con.execute(f"DROP TABLE IF EXISTS {tbl}")

    con.close()
    return to_drop
