"""
Startup script — run this once when you open VS Code.

1. Refreshes the warehouse from inbox/
2. Watches inbox/ for new files while you work

Usage:
    python "HS Data/start.py"
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from refresh import refresh
from watcher import InboxHandler, INBOX, SUPPORTED
from watchdog.observers import Observer

# Step 1: Refresh
try:
    refresh()
except Exception as e:
    if "already open" in str(e) or "IO Error" in str(e):
        print("\nwarehouse.duckdb is locked by the Database Client extension.")
        print("In VS Code: right-click your DuckDB connection -> Disconnect, then rerun this script.")
        raise SystemExit(1)
    raise

# Step 2: Watch
print(f"\nWatching inbox for new files. Press Ctrl+C to stop.\n")
INBOX.mkdir(exist_ok=True)
observer = Observer()
observer.schedule(InboxHandler(), str(INBOX), recursive=True)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopped.")
    observer.stop()
observer.join()
