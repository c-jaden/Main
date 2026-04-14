"""
Full refresh — drops all tables and reloads every file in inbox/.

Usage:
    python scripts/refresh.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from loader import load_file, connect, INBOX, SUPPORTED


def refresh():
    print("Dropping all existing tables...")
    con = connect()
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    for tbl in tables:
        con.execute(f"DROP TABLE IF EXISTS {tbl}")
    con.close()
    print(f"  Dropped {len(tables)} table(s).\n")

    files = sorted(f for f in INBOX.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED)

    if not files:
        print(f"No supported files found in {INBOX}")
        return

    print(f"Loading {len(files)} file(s) from inbox...\n")
    for path in files:
        print(f"  {path.name}")
        tables = load_file(path)
        for tbl in tables:
            print(f"    -> {tbl}")

    print("\nDone. Current tables:")
    con = connect()
    rows = con.execute("SHOW TABLES").fetchall()
    for (tbl,) in rows:
        count = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:40s} {count:>10,} rows")
    con.close()


if __name__ == "__main__":
    refresh()
