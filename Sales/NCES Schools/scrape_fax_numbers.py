"""
School Fax Number Scraper
==========================
Scrapes fax numbers from public school websites and stores results
in a separate fax_results table in the existing schools.db database.

Requires schools.db to already exist (run build_school_database.py first).

Usage:
    pip install requests beautifulsoup4 aiohttp tqdm
    python scrape_fax_numbers.py

    # Limit to a specific state:
    python scrape_fax_numbers.py --state OH

    # Resume a previously interrupted run:
    python scrape_fax_numbers.py --resume

    # Test with a small sample:
    python scrape_fax_numbers.py --limit 100

Output:
    Adds/updates fax_results table in schools.db with columns:
      nces_id, website, fax_number, confidence, confidence_reason,
      source_url, scraped_at
"""

import re
import time
import sqlite3
import argparse
import asyncio
import logging
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

DB_PATH         = "schools.db"
REQUEST_TIMEOUT = 5        # seconds before giving up on a page
MAX_CONCURRENCY = 20       # simultaneous requests
RATE_DELAY      = 0.1      # seconds between launching each request
MAX_CONTENT_KB  = 500      # skip pages larger than this (KB)

# Contact page path hints — tried in order before falling back to homepage
CONTACT_PATHS = [
    "/contact", "/contact-us", "/contactus",
    "/about/contact", "/about-us/contact",
    "/school-info", "/information",
]

# ── Fax number patterns ───────────────────────────────────────────────────────

# Matches common US phone number formats
PHONE_RE = re.compile(
    r"(\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4})"
)

# "Fax" label within 120 characters before a phone number
FAX_LABEL_RE = re.compile(
    r"(?i)fax[:\s#\.\-]{0,5}[\s]*"
    r"(\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4})"
)

# Normalize a phone number to digits only for comparison
def normalize_phone(p: str) -> str:
    return re.sub(r"\D", "", p)


# ── Confidence scoring ────────────────────────────────────────────────────────

def score_fax(
    fax_raw: str,
    school_phone: str,
    labeled: bool,
    on_contact_page: bool,
    count_on_page: int,
) -> tuple[str, str]:
    """
    Returns (confidence, reason) where confidence is High / Medium / Low.
    """
    fax_digits   = normalize_phone(fax_raw)
    phone_digits = normalize_phone(school_phone or "")

    reasons = []

    # Disqualifiers
    if fax_digits == phone_digits and phone_digits:
        return "Low", "Fax matches school phone number — likely same line"

    if len(fax_digits) != 10:
        return "Low", "Number does not appear to be a valid 10-digit US number"

    # Positive signals
    score = 0

    if labeled:
        score += 3
        reasons.append("explicitly labeled as Fax")

    if on_contact_page:
        score += 2
        reasons.append("found on contact page")

    if phone_digits and fax_digits[:3] == phone_digits[:3]:
        score += 1
        reasons.append("same area code as school phone")

    if count_on_page == 1:
        score += 1
        reasons.append("only one fax number on page")
    elif count_on_page > 3:
        score -= 1
        reasons.append("many numbers on page — ambiguous")

    if score >= 5:
        confidence = "High"
    elif score >= 2:
        confidence = "Medium"
    else:
        confidence = "Low"

    reason = "; ".join(reasons) if reasons else "found via pattern match only"
    return confidence, reason


# ── HTML parsing ──────────────────────────────────────────────────────────────

def extract_fax_from_html(html: str, school_phone: str,
                           on_contact_page: bool, source_url: str) -> list[dict]:
    """
    Parse HTML and return list of fax candidates with scoring.
    Each entry: {fax_number, confidence, confidence_reason, source_url}
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style noise
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    # 1. Look for explicitly labeled fax numbers (highest signal)
    labeled_matches = FAX_LABEL_RE.findall(text)

    # 2. Look for all phone-like numbers on the page for context
    all_phones = PHONE_RE.findall(text)
    labeled_normalized = {normalize_phone(m) for m in labeled_matches}

    results = []

    # Score labeled fax numbers first
    for raw in labeled_matches:
        conf, reason = score_fax(
            raw, school_phone,
            labeled=True,
            on_contact_page=on_contact_page,
            count_on_page=len(labeled_matches),
        )
        results.append({
            "fax_number":        raw.strip(),
            "confidence":        conf,
            "confidence_reason": reason,
            "source_url":        source_url,
        })

    # If no labeled fax found, return nothing — unlabeled numbers are too noisy
    return results


# ── Async fetcher ─────────────────────────────────────────────────────────────

async def fetch(session: aiohttp.ClientSession, url: str) -> str | None:
    """Fetch a URL, return HTML text or None on failure."""
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            allow_redirects=True,
            ssl=False,
        ) as resp:
            if resp.status != 200:
                return None
            ct = resp.headers.get("Content-Type", "")
            if "html" not in ct:
                return None
            # Check content length before reading
            cl = resp.headers.get("Content-Length")
            if cl and int(cl) > MAX_CONTENT_KB * 1024:
                return None
            return await resp.text(errors="replace")
    except Exception:
        return None


def build_contact_urls(base_url: str) -> list[str]:
    """Return a list of candidate contact page URLs to try."""
    parsed = urlparse(base_url)
    root   = f"{parsed.scheme}://{parsed.netloc}"
    urls   = [urljoin(root, path) for path in CONTACT_PATHS]
    urls.append(base_url)   # homepage as final fallback
    return urls


async def scrape_school(
    session: aiohttp.ClientSession,
    nces_id: str,
    website: str,
    phone: str,
) -> dict:
    """
    Try contact pages first, then homepage.
    Returns best fax result dict or a no-result dict.
    """
    base_url = website.strip()
    if not base_url.startswith("http"):
        base_url = "https://" + base_url

    candidate_urls = build_contact_urls(base_url)
    all_results    = []

    for i, url in enumerate(candidate_urls):
        on_contact = i < len(CONTACT_PATHS)
        html = await fetch(session, url)
        if not html:
            continue
        found = extract_fax_from_html(html, phone, on_contact, url)
        all_results.extend(found)
        # If we found a High-confidence result, stop early
        if any(r["confidence"] == "High" for r in found):
            break

    if not all_results:
        return {
            "nces_id":           nces_id,
            "website":           website,
            "fax_number":        None,
            "confidence":        None,
            "confidence_reason": "No fax number found",
            "source_url":        None,
            "scraped_at":        datetime.now(timezone.utc).isoformat(),
        }

    # Pick best result: High > Medium > Low, then first found
    priority = {"High": 0, "Medium": 1, "Low": 2}
    best = sorted(all_results, key=lambda r: priority.get(r["confidence"], 9))[0]

    return {
        "nces_id":           nces_id,
        "website":           website,
        "fax_number":        best["fax_number"],
        "confidence":        best["confidence"],
        "confidence_reason": best["confidence_reason"],
        "source_url":        best["source_url"],
        "scraped_at":        datetime.now(timezone.utc).isoformat(),
    }


# ── Database helpers ──────────────────────────────────────────────────────────

def init_fax_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fax_results (
            nces_id           TEXT PRIMARY KEY,
            website           TEXT,
            fax_number        TEXT,
            confidence        TEXT,
            confidence_reason TEXT,
            source_url        TEXT,
            scraped_at        TEXT
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fax_confidence ON fax_results(confidence)"
    )
    conn.commit()


def load_schools(conn: sqlite3.Connection, state: str | None,
                 limit: int | None, resume: bool) -> list[dict]:
    """Load public schools with a WEBSITE value from schools.db."""
    where_clauses = [
        "school_type = 'public'",
        "website IS NOT NULL",
        "website != ''",
        "website != 'N'",
        "website != 'M'",
    ]
    if state:
        where_clauses.append(f"state_abbr = '{state.upper()}'")

    if resume:
        where_clauses.append(
            "nces_id NOT IN (SELECT nces_id FROM fax_results)"
        )

    sql = (
        "SELECT nces_id, website, phone "
        "FROM schools WHERE " + " AND ".join(where_clauses)
    )
    if limit:
        sql += f" LIMIT {limit}"

    rows = conn.execute(sql).fetchall()
    return [{"nces_id": r[0], "website": r[1], "phone": r[2]} for r in rows]


def save_results(conn: sqlite3.Connection, results: list[dict]):
    conn.executemany(
        """
        INSERT OR REPLACE INTO fax_results
            (nces_id, website, fax_number, confidence,
             confidence_reason, source_url, scraped_at)
        VALUES
            (:nces_id, :website, :fax_number, :confidence,
             :confidence_reason, :source_url, :scraped_at)
        """,
        results,
    )
    conn.commit()


# ── Main async loop ───────────────────────────────────────────────────────────

async def run(schools: list[dict], conn: sqlite3.Connection):
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    connector = aiohttp.TCPConnector(ssl=False, limit=MAX_CONCURRENCY)
    headers   = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; SchoolDirectoryBot/1.0; "
            "+https://nces.ed.gov)"
        )
    }

    async with aiohttp.ClientSession(
        connector=connector, headers=headers
    ) as session:

        async def bounded(school):
            async with sem:
                await asyncio.sleep(RATE_DELAY)
                return await scrape_school(
                    session,
                    school["nces_id"],
                    school["website"],
                    school.get("phone") or "",
                )

        tasks   = [bounded(s) for s in schools]
        batch   = []
        matched = 0

        with tqdm(total=len(tasks), desc="Scraping", unit="school") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                batch.append(result)
                if result["fax_number"]:
                    matched += 1
                pbar.set_postfix(fax_found=matched)
                pbar.update(1)

                # Write to DB every 200 results
                if len(batch) >= 200:
                    save_results(conn, batch)
                    batch = []

        if batch:
            save_results(conn, batch)

    return matched


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape fax numbers from school websites")
    parser.add_argument("--state",  type=str, default=None,
                        help="Only scrape schools in this state (e.g. OH)")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Only scrape this many schools (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip schools already in fax_results table")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    conn = sqlite3.connect(DB_PATH)
    init_fax_table(conn)

    schools = load_schools(conn, args.state, args.limit, args.resume)
    if not schools:
        print("No schools found to scrape. Check that schools.db exists and "
              "has public schools with WEBSITE values.")
        return

    print(f"\n=== School Fax Scraper ===")
    print(f"  Schools to scrape : {len(schools):,}")
    print(f"  Concurrency       : {MAX_CONCURRENCY} simultaneous requests")
    print(f"  Timeout per page  : {REQUEST_TIMEOUT}s")
    print(f"  State filter      : {args.state or 'All'}")
    print(f"  Resume mode       : {args.resume}\n")

    start   = time.time()
    matched = asyncio.run(run(schools, conn))
    elapsed = time.time() - start

    # ── Summary ──
    total_scraped = conn.execute("SELECT COUNT(*) FROM fax_results").fetchone()[0]
    by_conf = conn.execute(
        "SELECT confidence, COUNT(*) FROM fax_results "
        "WHERE fax_number IS NOT NULL GROUP BY confidence"
    ).fetchall()

    print(f"\n── Summary ──────────────────────────────────────")
    print(f"  Scraped this run  : {len(schools):,}")
    print(f"  Fax numbers found : {matched:,} ({matched/len(schools)*100:.1f}%)")
    print(f"  Elapsed           : {elapsed/60:.1f} min")
    print(f"  Total in DB       : {total_scraped:,}")
    print(f"\n  By confidence:")
    for conf, count in sorted(by_conf):
        print(f"    {conf:8s} : {count:,}")
    print("─────────────────────────────────────────────────\n")
    print("Example queries:")
    print("  -- All high-confidence fax numbers in Ohio")
    print("  SELECT s.name, s.city, f.fax_number, f.confidence")
    print("  FROM schools s JOIN fax_results f ON s.nces_id = f.nces_id")
    print("  WHERE s.state_abbr = 'OH' AND f.confidence = 'High';")

    conn.close()


if __name__ == "__main__":
    main()