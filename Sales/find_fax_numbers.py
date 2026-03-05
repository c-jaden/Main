"""
Scrape fax numbers from a list of school/district websites.

- Reads:  WA School Websites.csv (column: WEBSITE)
- Writes: wa_school_fax_numbers.csv
- Shows:  live progress bar in VS Code terminal (tqdm)
- Logs:   running counts + last site processed
"""

import re
import time
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


INPUT_CSV = "WA School Websites.csv"          # <-- your file
INPUT_COL = "WEBSITE"                         # <-- column name in your file
OUTPUT_CSV = "wa_school_fax_numbers.csv"

# --- Fax detection regexes (covers "Fax:", "F:", and common phone formatting) ---
FAX_LABEL_RE = re.compile(r"\b(fax|facsimile)\b", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?:(?:\+?1\s*[-\.\(]?\s*)?)"           # optional +1 / 1
    r"(?:\(\s*\d{3}\s*\)|\d{3})"             # area code
    r"\s*[-\.]?\s*"
    r"\d{3}"
    r"\s*[-\.]?\s*"
    r"\d{4}"
)

# Pages on school sites where fax numbers often live
KEYWORD_LINK_RE = re.compile(
    r"(contact|directory|staff|about|administration|office|communications|district|school\s*board)",
    re.IGNORECASE
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


def normalize_url(url: str) -> str:
    url = str(url).strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def same_domain(a: str, b: str) -> bool:
    try:
        return urlparse(a).netloc.lower().replace("www.", "") == urlparse(b).netloc.lower().replace("www.", "")
    except Exception:
        return False


def fetch_html(url: str, timeout=20) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    ct = (r.headers.get("Content-Type") or "").lower()
    # Some servers omit content-type; treat empty as potentially HTML and let parsing handle it.
    if ct and ("text/html" not in ct and "application/xhtml+xml" not in ct):
        return ""
    return r.text


def extract_faxes_from_text(text: str):
    """
    Returns list of dicts: {"fax": "...", "context": "..."}
    Strategy:
      - find phone-like numbers
      - prefer ones near "fax" labels
      - if the page mentions fax at all, accept phone numbers (lower bar) since formatting varies
    """
    if not text:
        return []

    collapsed = re.sub(r"\s+", " ", text)

    candidates = []
    for m in PHONE_RE.finditer(collapsed):
        num = m.group(0).strip()

        # context window around the number
        start = max(0, m.start() - 40)
        end = min(len(collapsed), m.end() + 40)
        context = collapsed[start:end]

        score = 1
        if FAX_LABEL_RE.search(context):
            score += 10

        wide_start = max(0, m.start() - 120)
        wide_end = min(len(collapsed), m.end() + 120)
        if FAX_LABEL_RE.search(collapsed[wide_start:wide_end]):
            score += 5

        candidates.append((score, num, context))

    candidates.sort(reverse=True, key=lambda x: x[0])

    seen = set()
    results = []
    page_mentions_fax = bool(FAX_LABEL_RE.search(collapsed))

    for score, num, context in candidates:
        key = re.sub(r"\D", "", num)
        if len(key) < 10:
            continue
        if key in seen:
            continue
        seen.add(key)

        # Strong accept if "fax" label nearby OR page mentions fax somewhere
        if score >= 11 or page_mentions_fax:
            results.append({"fax": num, "context": context.strip()})

    return results


def extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    return ""


def get_candidate_links(base_url: str, soup: BeautifulSoup, limit=6):
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(" ", strip=True)

        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue

        full = urljoin(base_url, href)
        if not same_domain(base_url, full):
            continue

        if KEYWORD_LINK_RE.search(href) or KEYWORD_LINK_RE.search(text):
            links.append(full)

    # de-dupe while preserving order
    deduped = []
    seen = set()
    for u in links:
        if u not in seen:
            seen.add(u)
            deduped.append(u)

    return deduped[:limit]


def process_site(url: str, sleep_s=0.5):
    """
    Returns: (site_title, uniq_faxes, sources, errors)
      - uniq_faxes: list of {"fax": str, "context": str}
      - sources: list[str] pages where fax was found
      - errors: list[str]
    """
    errors = []
    found = []
    sources = []

    url = normalize_url(url)
    if not url:
        return "", [], [], ["Empty URL"]

    # Try homepage
    try:
        html = fetch_html(url)
        if not html:
            errors.append("Non-HTML or empty content at homepage")
            return "", [], [], errors

        soup = BeautifulSoup(html, "html.parser")
        title = extract_title(soup)

        text = soup.get_text(" ", strip=True)
        faxes = extract_faxes_from_text(text)
        if faxes:
            found.extend(faxes)
            sources.append(url)

        # Crawl likely pages
        candidate_links = get_candidate_links(url, soup, limit=6)

        # Add common paths (many school CMS systems support these)
        for path in ["/contact", "/contact-us", "/contactus", "/directory", "/staff", "/about"]:
            candidate_links.append(urljoin(url, path))

        # De-dupe
        seen = set()
        final_links = []
        for l in candidate_links:
            if l not in seen:
                seen.add(l)
                final_links.append(l)

        # Visit up to 10 candidate pages
        for link in final_links[:10]:
            try:
                time.sleep(sleep_s)
                html2 = fetch_html(link)
                if not html2:
                    continue
                soup2 = BeautifulSoup(html2, "html.parser")
                text2 = soup2.get_text(" ", strip=True)
                faxes2 = extract_faxes_from_text(text2)
                if faxes2:
                    found.extend(faxes2)
                    sources.append(link)
            except Exception as e:
                errors.append(f"Error fetching {link}: {type(e).__name__}")
                continue

        # Final de-dupe fax list
        uniq = []
        seen_nums = set()
        for item in found:
            key = re.sub(r"\D", "", item["fax"])
            if key not in seen_nums:
                seen_nums.add(key)
                uniq.append(item)

        return title, uniq, sources, errors

    except Exception as e:
        errors.append(f"Homepage error: {type(e).__name__}: {e}")
        return "", [], [], errors


def main():
    df = pd.read_csv(INPUT_CSV)
    if INPUT_COL not in df.columns:
        raise ValueError(f"Column '{INPUT_COL}' not found. Found columns: {list(df.columns)}")

    urls = df[INPUT_COL].astype(str).tolist()
    total = len(urls)

    rows = []
    found_count = 0
    not_found_count = 0
    error_count = 0

    pbar = tqdm(urls, total=total, desc="Processing Schools", unit="site")
    for i, raw_url in enumerate(pbar, start=1):
        url = normalize_url(raw_url)

        # update what VS Code terminal shows "right now"
        pbar.set_postfix_str(urlparse(url).netloc[:40] if url else "empty-url")

        title, faxes, sources, errors = process_site(url)

        if faxes:
            fax_join = "; ".join([f["fax"] for f in faxes])
            found_count += 1
        else:
            fax_join = "Not found"
            not_found_count += 1

        if errors:
            error_count += 1

        rows.append({
            "website": url,
            "site_title": title,
            "fax_numbers": fax_join,
            "found_on": "; ".join(sources) if sources else "",
            "errors": "; ".join(errors) if errors else ""
        })

        # occasional detailed log lines that don't break tqdm
        if i % 25 == 0 or i == total:
            tqdm.write(
                f"[{i}/{total}] Found: {found_count} | Not found: {not_found_count} | Sites w/ errors: {error_count}"
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Wrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    """
    Run:
      pip install pandas requests beautifulsoup4 tqdm
      python scrape_faxes.py
    """
    main()
