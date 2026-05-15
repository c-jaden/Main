"""
File watcher — drop a file into inbox/ and it loads automatically.
If the DB is locked by the extension, the file is queued and retried every 15s.

Usage:
    python scripts/watcher.py

Keep this running in a terminal while you work. Press Ctrl+C to stop.
"""

import sys
import time
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sys.path.insert(0, str(Path(__file__).parent))
from loader import load_file, drop_file, INBOX, SUPPORTED

import duckdb

_retry_queue: list[tuple[str, Path]] = []  # list of ("created"|"modified"|"deleted", path)
_queue_lock = threading.Lock()


def _is_locked() -> bool:
    try:
        from loader import DB, connect
        con = connect(retries=1, delay=0)
        con.close()
        return False
    except duckdb.IOException:
        return True


def _try_load(action: str, path: Path) -> bool:
    """Attempt to process a file. Returns True on success, False if DB is locked."""
    try:
        if action in ("created", "modified"):
            tables = load_file(path)
            for tbl in tables:
                print(f"  -> {tbl}")
        elif action == "deleted":
            dropped = drop_file(path)
            for tbl in dropped:
                print(f"  dropped: {tbl}")
        return True
    except duckdb.IOException:
        return False


def _retry_worker():
    """Background thread: flush the retry queue whenever the DB becomes available."""
    while True:
        time.sleep(15)
        with _queue_lock:
            if not _retry_queue:
                continue
            if _is_locked():
                print(f"  [retry] DB still locked, {len(_retry_queue)} file(s) pending...")
                continue
            remaining = []
            for action, path in _retry_queue:
                print(f"\n[retry] {path.name}")
                if not _try_load(action, path):
                    remaining.append((action, path))
            _retry_queue[:] = remaining


def _handle(action: str, path: Path):
    if not _try_load(action, path):
        print(f"  DB locked — queued for retry when extension disconnects")
        with _queue_lock:
            _retry_queue.append((action, path))


class InboxHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in SUPPORTED:
            return
        print(f"\n+ {path.name}")
        _handle("created", path)

    def on_modified(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in SUPPORTED:
            return
        print(f"\n~ {path.name} (updated)")
        _handle("modified", path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in SUPPORTED:
            return
        print(f"\n- {path.name} (removed)")
        _handle("deleted", path)


if __name__ == "__main__":
    INBOX.mkdir(exist_ok=True)
    print(f"Watching: {INBOX}")
    print(f"Supported: {', '.join(SUPPORTED)}")
    print("Press Ctrl+C to stop.\n")

    threading.Thread(target=_retry_worker, daemon=True).start()

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
