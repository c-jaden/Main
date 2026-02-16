#!/usr/bin/env python3
"""
Find fax numbers on a list of websites.

Upgrades:
- Extracts numbers from tel: links (common in site footers)
- Adds "Find Us" heuristic: if a Find Us/address block has 2 numbers, assume 2nd is fax
- Still prioritizes explicit "Fax:" labeling and schema.org JSON-LD fax fields

Usage:
  python find_fax_numbers.py --input sites.txt --output fax_results.csv
  python find_fax_numbers.py --sites https://example.com https://example.org

Dependencies:
  pip install requests beautifulsoup4 lxml
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup


# ----------------------------
# Config
# ----------------------------

DEFAULT_PATH_HINTS = [
    "/contact",
    "/contact-us",
    "/contacts",
    "/about",
    "/about-us",
    "/support",
    "/help",
    "/customer-service",
    "/locations",
    "/district",
    "/our-district",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FaxFinder/1.1; +https://example.com/bot)"
}

PHONE_CANDIDATE_RE = re.compile(
    r"""
    (?:
        (?:\+?\d{1,3}[\s\-.()]*)?          # optional country code
        (?:\(?\d{2,4}\)?[\s\-.()]*)       # area code
        \d{2,4}[\s\-.()]*\d{3,4}          # local number
        (?:\s*(?:x|ext\.?)\s*\d{1,6})?    # optional extension
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

FAX_KEYWORDS_RE = re.compile(r"\bfax\b|\bfaks\b|\btélécop(ieur|ie)\b", re.IGNORECASE)

# "Find us" patterns (common in school CMS footers)
FIND_US_RE = re.compile(r"\bfind us\b|\bvisit us\b|\bcontact us\b|\blocation\b|\baddress\b", re.IGNORECASE)

SCHEMA_FAX_KEYS = {"faxNumber", "fax", "fax_number", "faxnumber"}


@dataclass
class FaxHit:
    fax: str
    score: float
    page_url: str
    context: str


# ----------------------------
# Helpers
# ----------------------------

def normalize_url(u: str) -> str:
    u = u.strip()
    if not u:
        return u
    if not re.match(r"^https?://", u, re.IGNORECASE):
        u = "https://" + u
    return u


def same_domain(base: str, target: str) -> bool:
    b = urlparse(base).netloc.lower()
    t = urlparse(target).netloc.lower()
    return b == t or t.endswith("." + b)


def clean_phone(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"\s+", " ", raw)
    return raw


def digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def get_text_and_soup(html: str) -> Tuple[str, BeautifulSoup]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return text, soup


def fetch(session: requests.Session, url: str, timeout: int = 20) -> Optional[str]:
    try:
        r = session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code >= 400:
            return None
        r.encoding = r.apparent_encoding or r.encoding
        return r.text
    except requests.RequestException:
        return None


def extract_schema_fax(soup: BeautifulSoup) -> List[str]:
    results: List[str] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "")
        except Exception:
            continue

        def walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if str(k) in SCHEMA_FAX_KEYS and isinstance(v, str):
                        results.append(clean_phone(v))
                    walk(v)
            elif isinstance(obj, list):
                for it in obj:
                    walk(it)

        walk(data)

    # dedupe
    seen = set()
    out = []
    for x in results:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def find_contact_like_links(base_url: str, soup: BeautifulSoup, max_links: int = 8) -> List[str]:
    candidates = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        abs_url = urljoin(base_url, href)
        if not abs_url.lower().startswith(("http://", "https://")):
            continue
        if not same_domain(base_url, abs_url):
            continue

        text = (a.get_text(" ", strip=True) or "").lower()
        href_l = abs_url.lower()
        if any(k in text for k in ["contact", "support", "locations", "about", "district"]) or any(
            k in href_l for k in ["contact", "support", "location", "about", "help", "district"]
        ):
            candidates.append(abs_url)

    seen = set()
    out = []
    for u in candidates:
        if u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= max_links:
            break
    return out


def extract_tel_links(soup: BeautifulSoup) -> List[str]:
    """Extract numbers from <a href="tel:..."> links."""
    nums: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href.lower().startswith("tel:"):
            continue
        val = href[4:]
        val = unquote(val)
        # strip params like tel:+1...;ext=123
        val = val.split(";")[0].split("?")[0]
        val = val.strip()
        if val:
            nums.append(clean_phone(val))

    # dedupe
    seen = set()
    out = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def score_fax_candidates(text: str, page_url: str) -> List[FaxHit]:
    hits: List[FaxHit] = []

    # 1) Explicit "Fax: <number>" (highest confidence)
    explicit_re = re.compile(
        r"(fax[^0-9]{0,15})(" + PHONE_CANDIDATE_RE.pattern + r")",
        re.IGNORECASE | re.VERBOSE,
    )
    for m in explicit_re.finditer(text):
        context = text[max(0, m.start() - 60): m.end() + 60]
        fax = clean_phone(m.group(2))
        if len(digits_only(fax)) >= 7:
            hits.append(FaxHit(fax=fax, score=95.0, page_url=page_url, context=context))

    # 2) Generic candidates scored by proximity to "fax"
    keywords = [m.start() for m in FAX_KEYWORDS_RE.finditer(text)]
    for m in PHONE_CANDIDATE_RE.finditer(text):
        fax = clean_phone(m.group(0))
        if len(digits_only(fax)) < 7:
            continue

        score = 10.0
        if keywords:
            d = min(abs(m.start() - k) for k in keywords)
            if d <= 10:
                score = 90.0
            elif d <= 30:
                score = 70.0
            elif d <= 80:
                score = 45.0
            elif d <= 150:
                score = 25.0

        snippet = text[max(0, m.start() - 40): m.end() + 40]
        if FAX_KEYWORDS_RE.search(snippet):
            score = max(score, 80.0)

        if score >= 25.0:
            hits.append(FaxHit(fax=fax, score=score, page_url=page_url, context=snippet))

    # Deduplicate by fax value
    best: dict[str, FaxHit] = {}
    for h in hits:
        if h.fax not in best or h.score > best[h.fax].score:
            best[h.fax] = h

    return sorted(best.values(), key=lambda x: x.score, reverse=True)


def find_findus_block_fax(text: str, page_url: str) -> Optional[FaxHit]:
    """
    Heuristic:
    - Find a window of text around "Find Us"/address-like keywords
    - If the window contains >= 2 phone-like numbers and none are explicitly labeled "fax",
      assume the SECOND number is the fax.
    """
    # If "fax" appears anywhere, don’t apply this heuristic (we already have better logic)
    if FAX_KEYWORDS_RE.search(text):
        return None

    m = FIND_US_RE.search(text)
    if not m:
        return None

    # Grab a window around the match (footer blocks often get flattened into one line)
    start = max(0, m.start() - 250)
    end = min(len(text), m.end() + 400)
    window = text[start:end]

    nums = [clean_phone(x.group(0)) for x in PHONE_CANDIDATE_RE.finditer(window)]
    # Filter to plausible numbers
    nums = [n for n in nums if len(digits_only(n)) >= 7]

    if len(nums) >= 2:
        fax = nums[1]
        return FaxHit(
            fax=fax,
            score=55.0,  # medium confidence (heuristic)
            page_url=page_url,
            context=f'Find-Us heuristic window: "{window[:200]}..."',
        )

    return None


def choose_best(hits: List[FaxHit]) -> Optional[FaxHit]:
    return hits[0] if hits else None


def scan_page_for_fax(page_url: str, html: str) -> List[FaxHit]:
    text, soup = get_text_and_soup(html)
    hits: List[FaxHit] = []

    # Schema fax (very high confidence)
    for f in extract_schema_fax(soup):
        if len(digits_only(f)) >= 7:
            hits.append(FaxHit(fax=f, score=98.0, page_url=page_url, context="schema.org JSON-LD"))

    # tel: links (often contain the same numbers shown visually)
    for n in extract_tel_links(soup):
        # Score tel links modestly; if fax is explicit nearby, the text scoring will win anyway
        if len(digits_only(n)) >= 7:
            hits.append(FaxHit(fax=n, score=20.0, page_url=page_url, context="tel: link"))

    # Normal text scoring
    hits.extend(score_fax_candidates(text, page_url=page_url))

    # Find-us heuristic
    heuristic_hit = find_findus_block_fax(text, page_url=page_url)
    if heuristic_hit:
        hits.append(heuristic_hit)

    # Deduplicate, keep best per number
    best: dict[str, FaxHit] = {}
    for h in hits:
        if h.fax not in best or h.score > best[h.fax].score:
            best[h.fax] = h

    return sorted(best.values(), key=lambda x: x.score, reverse=True)


def scan_site_for_fax(site: str, sleep_s: float = 0.25) -> Tuple[Optional[FaxHit], List[FaxHit]]:
    site = normalize_url(site)
    session = requests.Session()

    html = fetch(session, site)
    if not html:
        return None, []

    text, soup = get_text_and_soup(html)

    all_hits: List[FaxHit] = []
    all_hits.extend(scan_page_for_fax(site, html))

    # Crawl likely pages
    urls_to_try = [urljoin(site, p) for p in DEFAULT_PATH_HINTS]
    urls_to_try += find_contact_like_links(site, soup, max_links=8)

    seen = {site}
    for u in urls_to_try:
        if u in seen:
            continue
        seen.add(u)
        time.sleep(sleep_s)
        sub_html = fetch(session, u)
        if not sub_html:
            continue
        all_hits.extend(scan_page_for_fax(u, sub_html))

    all_hits = sorted(all_hits, key=lambda x: x.score, reverse=True)
    best = choose_best(all_hits)
    return best, all_hits


# ----------------------------
# CLI
# ----------------------------

def read_sites_from_file(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="Text file with one site per line")
    ap.add_argument("--sites", nargs="*", help="Sites passed directly on CLI")
    ap.add_argument("--output", default="fax_results.csv", help="Output CSV path")
    ap.add_argument("--sleep", type=float, default=0.25, help="Delay between page requests (seconds)")
    args = ap.parse_args()

    sites: List[str] = []
    if args.input:
        sites.extend(read_sites_from_file(args.input))
    if args.sites:
        sites.extend(args.sites)

    if not sites:
        raise SystemExit("Provide --input sites.txt or --sites ...")

    rows = []
    for site in sites:
        best, all_hits = scan_site_for_fax(site, sleep_s=args.sleep)
        if best:
            rows.append({
                "site": site,
                "fax": best.fax,
                "confidence": round(best.score, 1),
                "source_url": best.page_url,
                "context": (best.context or "")[:200],
            })
        else:
            rows.append({
                "site": site,
                "fax": "",
                "confidence": 0,
                "source_url": "",
                "context": "No fax found",
            })

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["site", "fax", "confidence", "source_url", "context"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
