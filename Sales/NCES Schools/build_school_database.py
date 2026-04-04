"""
NCES School Database Builder
==============================
Downloads public and private school data from NCES and combines
them into a single SQLite database (schools.db).

County + FIPS enrichment strategy (all free, no API key required):
  1. NCES data     — use existing county field where present
  2. Address batch — Census batch geocoder (up to 10,000 rows/request, fast)
  3. Lat/lon       — Census coordinate lookup for any remaining gaps

Usage:
    pip install requests pandas openpyxl tqdm
    python build_school_database.py

    # Skip all geocoding (use NCES county data as-is, fastest):
    python build_school_database.py --no-geocode

    # Skip address batch, only use lat/lon fallback:
    python build_school_database.py --no-address-batch

    # Enable debug mode (saves sent/received files, prints diagnostics):
    python build_school_database.py --debug

    # Debug a small sample only (fast, good for diagnosing issues):
    python build_school_database.py --debug --debug-rows 50

Output:
    schools.db              — SQLite database with all schools
    schools.csv             — Flat CSV export of all schools
    debug_sent_batch_N.csv  — (debug mode) address CSV sent to Census
    debug_recv_batch_N.csv  — (debug mode) raw response from Census
    debug_nces_county.csv   — (debug mode) sample of raw NCES county field values
"""

import os
import io
import time
import zipfile
import sqlite3
import argparse
import requests
import pandas as pd
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

PUBLIC_URL  = "https://nces.ed.gov/ccd/Data/zip/ccd_sch_029_2425_w_1a_073025.zip"
PRIVATE_URL = "https://nces.ed.gov/surveys/pss/zip/pss2122_pu_csv.zip"

OUTPUT_DIR = "."
DB_PATH    = os.path.join(OUTPUT_DIR, "schools.db")
CSV_PATH   = os.path.join(OUTPUT_DIR, "schools.csv")

ADDRESS_BATCH_SIZE = 2000   # Reduced — Census times out on large batches
COORD_BATCH_SIZE   = 500
CENSUS_RATE_DELAY  = 3.0
CENSUS_MAX_RETRIES = 3      # Retry failed batches up to this many times
CENSUS_RETRY_DELAY = 15.0   # Seconds to wait before retrying a failed batch

CENSUS_BATCH_URL  = "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch"
CENSUS_COORDS_URL = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"

# ── Diocese code → name lookup (Catholic schools only) ─────────────────────
# Source: NCES PSS 2021-22 User Manual
# Non-Catholic private schools will have diocese_code = 0 and diocese_name = ''
DIOCESE_TABLE = {
    "0101": "Diocese of Birmingham, AL",
    "0102": "Diocese of Mobile, AL",
    "0201": "Archdiocese of Anchorage, AK",
    "0202": "Diocese of Fairbanks, AK",
    "0203": "Diocese of Juneau, AK",
    "0401": "Diocese of Phoenix, AZ",
    "0402": "Diocese of Tucson, AZ",
    "0501": "Diocese of Little Rock, AR",
    "0601": "Archdiocese of Los Angeles, CA",
    "0602": "Archdiocese of San Francisco, CA",
    "0603": "Diocese of Fresno, CA",
    "0604": "Diocese of Monterey, CA",
    "0605": "Diocese of Oakland, CA",
    "0606": "Diocese of Orange, CA",
    "0607": "Diocese of Sacramento, CA",
    "0608": "Diocese of San Bernardino, CA",
    "0609": "Diocese of San Diego, CA",
    "0610": "Diocese of San Jose, CA",
    "0611": "Diocese of Santa Rosa, CA",
    "0612": "Diocese of Stockton, CA",
    "0801": "Archdiocese of Denver, CO",
    "0802": "Diocese of Colorado Springs, CO",
    "0803": "Diocese of Pueblo, CO",
    "0901": "Archdiocese of Hartford, CT",
    "0902": "Diocese of Bridgeport, CT",
    "0903": "Diocese of Norwich, CT",
    "1001": "Diocese of Wilmington, DE",
    "1101": "Archdiocese of Washington, DC",
    "1201": "Archdiocese of Miami, FL",
    "1202": "Diocese of Pensacola-Tallahassee, FL",
    "1203": "Diocese of Orlando, FL",
    "1204": "Diocese of Palm Beach, FL",
    "1205": "Diocese of St. Augustine, FL",
    "1206": "Diocese of St. Petersburg, FL",
    "1207": "Diocese of Venice, FL",
    "1301": "Archdiocese of Atlanta, GA",
    "1302": "Diocese of Savannah, GA",
    "1501": "Diocese of Honolulu, HI",
    "1601": "Diocese of Boise, ID",
    "1701": "Archdiocese of Chicago, IL",
    "1702": "Diocese of Belleville, IL",
    "1703": "Diocese of Joliet, IL",
    "1704": "Diocese of Peoria, IL",
    "1705": "Diocese of Rockford, IL",
    "1706": "Diocese of Springfield, IL",
    "1801": "Archdiocese of Indianapolis, IN",
    "1802": "Diocese of Evansville, IN",
    "1803": "Diocese of Ft. Wayne-South Bend, IN",
    "1804": "Diocese of Gary, IN",
    "1805": "Diocese of Lafayette, IN",
    "1901": "Archdiocese of Dubuque, IA",
    "1902": "Diocese of Davenport, IA",
    "1903": "Diocese of Des Moines, IA",
    "1904": "Diocese of Sioux City, IA",
    "2001": "Archdiocese of Kansas City, KS",
    "2002": "Diocese of Dodge City, KS",
    "2003": "Diocese of Salina, KS",
    "2004": "Diocese of Wichita, KS",
    "2101": "Archdiocese of Louisville, KY",
    "2102": "Diocese of Covington, KY",
    "2103": "Diocese of Lexington, KY",
    "2104": "Diocese of Owensboro, KY",
    "2201": "Archdiocese of New Orleans, LA",
    "2202": "Diocese of Alexandria, LA",
    "2203": "Diocese of Baton Rouge, LA",
    "2204": "Diocese of Houma-Thibodaux, LA",
    "2205": "Diocese of Lafayette, LA",
    "2206": "Diocese of Lake Charles, LA",
    "2207": "Diocese of Shreveport, LA",
    "2301": "Diocese of Portland, ME",
    "2401": "Archdiocese of Baltimore, MD",
    "2501": "Archdiocese of Boston, MA",
    "2502": "Diocese of Fall River, MA",
    "2503": "Diocese of Springfield, MA",
    "2504": "Diocese of Worcester, MA",
    "2601": "Archdiocese of Detroit, MI",
    "2602": "Diocese of Grand Rapids, MI",
    "2603": "Diocese of Gaylord, MI",
    "2604": "Diocese of Kalamazoo, MI",
    "2605": "Diocese of Lansing, MI",
    "2606": "Diocese of Marquette, MI",
    "2607": "Diocese of Saginaw, MI",
    "2701": "Archdiocese of St. Paul-Minneapolis, MN",
    "2702": "Diocese of Crookston, MN",
    "2703": "Diocese of Duluth, MN",
    "2704": "Diocese of New Ulm, MN",
    "2705": "Diocese of St. Cloud, MN",
    "2706": "Diocese of Winona, MN",
    "2801": "Diocese of Biloxi, MS",
    "2802": "Diocese of Jackson, MS",
    "2901": "Archdiocese of St. Louis, MO",
    "2902": "Diocese of Jefferson City, MO",
    "2903": "Diocese of Kansas City-St. Joseph, MO",
    "2904": "Diocese of Springfield-Cape Girardeau, MO",
    "3001": "Diocese of Great Falls-Billings, MT",
    "3002": "Diocese of Helena, MT",
    "3101": "Archdiocese of Omaha, NE",
    "3102": "Diocese of Grand Island, NE",
    "3103": "Diocese of Lincoln, NE",
    "3201": "Diocese of Las Vegas, NV",
    "3202": "Diocese of Reno, NV",
    "3301": "Diocese of Manchester, NH",
    "3401": "Archdiocese of Newark, NJ",
    "3402": "Diocese of Camden, NJ",
    "3403": "Diocese of Metuchen, NJ",
    "3404": "Diocese of Paterson, NJ",
    "3405": "Diocese of Trenton, NJ",
    "3501": "Archdiocese of Santa Fe, NM",
    "3502": "Diocese of Gallup, NM",
    "3503": "Diocese of Las Cruces, NM",
    "3601": "Archdiocese of New York, NY",
    "3602": "Diocese of Albany, NY",
    "3603": "Diocese of Brooklyn, NY",
    "3604": "Diocese of Buffalo, NY",
    "3605": "Diocese of Ogdensburg, NY",
    "3606": "Diocese of Rochester, NY",
    "3607": "Diocese of Rockville Centre, NY",
    "3608": "Diocese of Syracuse, NY",
    "3701": "Diocese of Charlotte, NC",
    "3702": "Diocese of Raleigh, NC",
    "3801": "Diocese of Bismarck, ND",
    "3802": "Diocese of Fargo, ND",
    "3901": "Archdiocese of Cincinnati, OH",
    "3902": "Diocese of Cleveland, OH",
    "3903": "Diocese of Columbus, OH",
    "3904": "Diocese of Steubenville, OH",
    "3905": "Diocese of Toledo, OH",
    "3906": "Diocese of Youngstown, OH",
    "4001": "Archdiocese of Oklahoma City, OK",
    "4002": "Diocese of Tulsa, OK",
    "4101": "Archdiocese of Portland, OR",
    "4102": "Diocese of Baker, OR",
    "4201": "Archdiocese of Philadelphia, PA",
    "4202": "Diocese of Allentown, PA",
    "4203": "Diocese of Altoona-Johnstown, PA",
    "4204": "Diocese of Erie, PA",
    "4205": "Diocese of Greensburg, PA",
    "4206": "Diocese of Harrisburg, PA",
    "4207": "Diocese of Pittsburgh, PA",
    "4208": "Diocese of Scranton, PA",
    "4401": "Diocese of Providence, RI",
    "4501": "Diocese of Charleston, SC",
    "4601": "Diocese of Rapid City, SD",
    "4602": "Diocese of Sioux Falls, SD",
    "4701": "Diocese of Knoxville, TN",
    "4702": "Diocese of Memphis, TN",
    "4703": "Diocese of Nashville, TN",
    "4801": "Archdiocese of San Antonio, TX",
    "4802": "Diocese of Amarillo, TX",
    "4803": "Diocese of Austin, TX",
    "4804": "Diocese of Beaumont, TX",
    "4805": "Diocese of Brownsville, TX",
    "4806": "Diocese of Corpus Christi, TX",
    "4807": "Diocese of Dallas, TX",
    "4808": "Diocese of El Paso, TX",
    "4809": "Diocese of Ft. Worth, TX",
    "4810": "Diocese of Galveston-Houston, TX",
    "4811": "Diocese of Lubbock, TX",
    "4812": "Diocese of San Angelo, TX",
    "4813": "Diocese of Tyler, TX",
    "4814": "Diocese of Victoria, TX",
    "4815": "Diocese of Laredo, TX",
    "4901": "Diocese of Salt Lake, UT",
    "5001": "Diocese of Burlington, VT",
    "5101": "Diocese of Arlington, VA",
    "5102": "Diocese of Richmond, VA",
    "5301": "Archdiocese of Seattle, WA",
    "5302": "Diocese of Spokane, WA",
    "5303": "Diocese of Yakima, WA",
    "5401": "Diocese of Wheeling-Charleston, WV",
    "5501": "Archdiocese of Milwaukee, WI",
    "5502": "Diocese of Green Bay, WI",
    "5503": "Diocese of La Crosse, WI",
    "5504": "Diocese of Madison, WI",
    "5505": "Diocese of Superior, WI",
    "5601": "Diocese of Cheyenne, WY"
}

# NCES sentinel values that mean "missing"
NCES_BAD = {"N", "M", "-1", "-2", "†", "‡", "–", "nan", "none", ""}

# Global debug flag (set via --debug CLI arg)
DEBUG      = False
DEBUG_ROWS = None   # if set, only geocode this many rows (for fast testing)

# ── Column mappings ───────────────────────────────────────────────────────────

PUBLIC_COLS = {
    "NCESSCH":   "nces_id",
    "SCH_NAME":  "name",
    "LSTREET1":  "address",
    "LCITY":     "city",
    "LZIP":      "zip",
    "ST":        "state_abbr",
    "PHONE":     "phone",
    "GSLO":      "grade_low",
    "GSHI":      "grade_high",
    "SCH_TYPE":  "school_type_code",
    "LEVEL":     "level",
    "FIPST":     "state_fips",
    "LEAID":     "district_id",
    "LEA_NAME":  "district_name",
    "WEBSITE":   "website",
}

PRIVATE_COLS = {
    "PPIN":        "nces_id",
    "PINST":       "name",
    "PADDRS":      "address",
    "PCITY":       "city",
    "PSTABB":      "state_abbr",
    "PZIP":        "zip",
    "PPHONE":      "phone",
    "LATITUDE22":  "latitude",
    "LONGITUDE22": "longitude",
    "LOGR2022":    "grade_low",
    "HIGR2022":    "grade_high",
    "NUMSTUDS":    "enrollment",
    "PCNTY":       "county",
    "DIOCESE":     "diocese_code",
}


# ── Download / parse helpers ──────────────────────────────────────────────────

def download_zip(url: str, label: str) -> dict:
    print(f"  Downloading {label}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    return {name: zf.read(name) for name in zf.namelist()}


def first_csv(file_dict: dict) -> pd.DataFrame:
    for name, data in file_dict.items():
        if name.lower().endswith(".csv"):
            print(f"  Parsing {name} ...")
            return pd.read_csv(
                io.BytesIO(data), encoding="latin-1",
                low_memory=False, dtype=str,
            )
    raise FileNotFoundError("No CSV found in zip archive")


def remap(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    available = {k: v for k, v in col_map.items() if k in df.columns}
    missing   = [k for k in col_map if k not in df.columns]
    if missing:
        print(f"  ⚠  Columns not found (may vary by year): {missing}")
    return df[list(available.keys())].rename(columns=available)


def clean(df: pd.DataFrame, school_type: str) -> pd.DataFrame:
    df = df.copy()
    df["school_type"] = school_type

    for col in list(PUBLIC_COLS.values()) + ["school_type"]:
        if col not in df.columns:
            df[col] = None


    for col in ("latitude", "longitude", "enrollment"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["name", "address", "city", "zip", "phone", "county"]:
        if col in df.columns:
            df[col] = df[col].str.strip()

    return df[df["name"].notna() & (df["name"] != "")]


# ── Debug helpers ─────────────────────────────────────────────────────────────

def debug_save(filename: str, content: str):
    """Save a debug file to the output directory."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  📄 Debug file saved → {path}")


def debug_nces_county(df: pd.DataFrame, school_type: str, n: int = 30):
    """
    Print and save a sample of raw NCES county field values so you can
    see exactly what values are present before any cleaning.
    """
    if "county" not in df.columns:
        print(f"  ⚠  No 'county' column found in {school_type} data")
        return

    sample = df[["name", "address", "city", "state_abbr", "county"]].head(n)
    print(f"\n  ── Raw NCES county sample ({school_type}, first {n} rows) ──")
    print(sample.to_string(index=True))

    # Value distribution of the county field
    val_counts = df["county"].fillna("(null)").value_counts().head(20)
    print(f"\n  Top 20 county field values ({school_type}):")
    print(val_counts.to_string())

    # Save to file
    out = f"debug_nces_county_{school_type}.csv"
    df[["nces_id", "name", "address", "city", "state_abbr", "zip",
        "latitude", "longitude", "county"]].to_csv(
        os.path.join(OUTPUT_DIR, out), index=True
    )
    print(f"  📄 Full county debug data saved → {out}")


def debug_address_sample(df: pd.DataFrame, school_type: str, n: int = 20):
    """Print a sample of the address fields that will be sent to Census."""
    sample = df[["name", "address", "city", "state_abbr", "zip"]].head(n)
    print(f"\n  ── Address fields sample ({school_type}, first {n} rows) ──")
    print(sample.to_string(index=True))

    # Check for common problems
    no_addr  = (df["address"].isna() | (df["address"] == "")).sum()
    no_city  = (df["city"].isna()    | (df["city"]    == "")).sum()
    no_state = df["state_abbr"].isna().sum()
    no_zip   = (df["zip"].isna()     | (df["zip"]     == "")).sum()
    po_box   = df["address"].str.upper().str.contains(
        r"P\.?O\.?\s*BOX|PO BOX", regex=True, na=False
    ).sum()
    rr_addr  = df["address"].str.upper().str.contains(
        r"^R\.?R\.?\s*\d|RURAL ROUTE|RR\s*\d", regex=True, na=False
    ).sum()

    print(f"\n  ── Address quality report ({school_type}) ──")
    print(f"  Total rows         : {len(df):,}")
    print(f"  Missing address    : {no_addr:,}")
    print(f"  Missing city       : {no_city:,}")
    print(f"  Missing state      : {no_state:,}")
    print(f"  Missing zip        : {no_zip:,}")
    print(f"  PO Box addresses   : {po_box:,}  ← Census geocoder cannot match these")
    print(f"  Rural Route (RR)   : {rr_addr:,}  ← Census geocoder may struggle")


# ── Census address batch geocoder (PRIMARY) ───────────────────────────────────

def _build_address_csv(chunk: pd.DataFrame) -> str:
    """
    Build the CSV payload for the Census batch geocoder.
    Format per row: Unique_ID,Street_Address,City,State,ZIP

    Uses csv.writer so fields with commas are properly quoted,
    preventing column-shift errors in the Census response.
    """
    import csv
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    for idx, row in chunk.iterrows():
        uid  = str(idx)
        addr = str(row.get("address",    "") or "").strip()
        city = str(row.get("city",       "") or "").strip()
        st   = str(row.get("state_abbr", "") or "").strip()
        zip_ = str(row.get("zip",        "") or "").strip()
        writer.writerow([uid, addr, city, st, zip_])
    return buf.getvalue()


def _parse_batch_response(text: str, batch_num: int) -> dict:
    """
    Parse the Census batch geocoder CSV response.
    Returns dict of {original_row_id: {county_fips, state_fips}}

    Census response columns (when matched):
      0  Unique_ID
      1  Input_Address
      2  Match_Status   (Match / No_Match / Tie)
      3  Match_Type     (Exact / Non_Exact)
      4  Matched_Address
      5  Coordinates    (lon,lat)
      6  TIGER_Line_ID
      7  Side
      8  State_FIPS     (2-digit)
      9  County_FIPS    (3-digit)
      10 Tract
      11 Block
    """
    if DEBUG:
        debug_save(f"debug_recv_batch_{batch_num}.csv", text)

    results     = {}
    total_lines = 0
    matched     = 0
    no_match    = 0
    bad_format  = 0

    for line in text.strip().splitlines():
        if not line.strip():
            continue
        total_lines += 1
        import csv as _csv
        parts = next(_csv.reader([line]))

        if len(parts) < 4:
            bad_format += 1
            if DEBUG and bad_format <= 5:
                print(f"  ⚠  Short line ({len(parts)} cols): {line[:120]}")
            continue

        uid        = parts[0].strip()
        match_stat = parts[2].strip().lower()

        if match_stat not in ("match", "tie"):
            no_match += 1
            if DEBUG and no_match <= 5:
                print(f"  No_Match row: {line[:120]}")
            continue

        state_fips = parts[8].strip().zfill(2) if len(parts) > 8 else ""
        county3    = parts[9].strip().zfill(3) if len(parts) > 9 else ""
        county_fips = state_fips + county3 if state_fips and county3 else ""

        if uid and county_fips:
            try:
                results[int(uid)] = {
                    "county_fips": county_fips,
                    "state_fips":  state_fips,
                    "county_name": "",
                }
                matched += 1
            except ValueError:
                bad_format += 1

    if DEBUG:
        print(f"  ── Batch {batch_num} parse report ──")
        print(f"     Total response lines : {total_lines:,}")
        print(f"     Matched              : {matched:,}")
        print(f"     No_Match             : {no_match:,}")
        print(f"     Bad format           : {bad_format:,}")
        if total_lines > 0:
            print(f"     Match rate           : {matched/total_lines*100:.1f}%")

    return results


def geocode_by_address(df: pd.DataFrame) -> dict:
    """
    Submit rows with a valid address to the Census batch geocoder.
    Returns dict of {row_index: {county_fips, state_fips, county_name}}.
    """
    has_address = (
        df["address"].notna() & (df["address"] != "") &
        df["city"].notna()    & (df["city"]    != "") &
        df["state_abbr"].notna()
    )
    target = df[has_address]

    # In debug mode, optionally limit to a small sample
    if DEBUG and DEBUG_ROWS:
        target = target.head(DEBUG_ROWS)
        print(f"  ⚙  Debug mode: limiting to first {DEBUG_ROWS} rows")

    indices = target.index.tolist()
    if not indices:
        print("  No rows with usable addresses — skipping address batch.")
        return {}

    batches = [
        indices[i : i + ADDRESS_BATCH_SIZE]
        for i in range(0, len(indices), ADDRESS_BATCH_SIZE)
    ]
    print(f"  Submitting {len(target):,} addresses in {len(batches)} batch(es)...")

    all_results = {}

    for batch_num, batch_idx in enumerate(
        tqdm(batches, desc="  Address batches", unit="batch"), start=1
    ):
        chunk    = target.loc[batch_idx]
        csv_data = _build_address_csv(chunk)

        if DEBUG:
            debug_save(f"debug_sent_batch_{batch_num}.csv", csv_data)
            print(f"\n  ── Batch {batch_num} send preview (first 5 rows) ──")
            for line in csv_data.splitlines()[:5]:
                print(f"     {line}")

        for attempt in range(1, CENSUS_MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    CENSUS_BATCH_URL,
                    files={"addressFile": ("addresses.csv",
                                           csv_data.encode("utf-8"),
                                           "text/csv")},
                    data={
                        "benchmark": "Public_AR_Current",
                        "vintage":   "Current_Current",
                    },
                    timeout=300,
                )

                if DEBUG:
                    print(f"  HTTP status: {resp.status_code}")
                    print(f"  Response size: {len(resp.text):,} chars")
                    print(f"  Response preview (first 5 lines):")
                    for line in resp.text.splitlines()[:5]:
                        print(f"     {line}")

                resp.raise_for_status()
                batch_results = _parse_batch_response(resp.text, batch_num)
                all_results.update(batch_results)
                break  # success — exit retry loop

            except requests.HTTPError as e:
                print(f"\n  ✗  HTTP {e.response.status_code} on batch {batch_num} "
                      f"(attempt {attempt}/{CENSUS_MAX_RETRIES})")
                if attempt < CENSUS_MAX_RETRIES:
                    print(f"     Retrying in {CENSUS_RETRY_DELAY}s...")
                    time.sleep(CENSUS_RETRY_DELAY)
                else:
                    print(f"     Giving up on batch {batch_num} after "
                          f"{CENSUS_MAX_RETRIES} attempts — will be missing from results")
            except Exception as e:
                print(f"\n  ⚠  Batch {batch_num} failed (attempt {attempt}/{CENSUS_MAX_RETRIES}): {e}")
                if attempt < CENSUS_MAX_RETRIES:
                    print(f"     Retrying in {CENSUS_RETRY_DELAY}s...")
                    time.sleep(CENSUS_RETRY_DELAY)
                else:
                    print(f"     Giving up on batch {batch_num}")

        if batch_num < len(batches):
            time.sleep(CENSUS_RATE_DELAY)

    print(f"  ✓  Address batch matched {len(all_results):,} / {len(target):,} schools")
    return all_results


# ── FIPS → county name lookup table ──────────────────────────────────────────

def fetch_fips_county_table() -> dict:
    url = (
        "https://www2.census.gov/geo/docs/reference/codes2020/"
        "national_county2020.txt"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), sep="|", dtype=str, header=0)
        df["fips5"] = df["STATEFP"].str.zfill(2) + df["COUNTYFP"].str.zfill(3)
        return dict(zip(df["fips5"], df["COUNTYNAME"]))
    except Exception as e:
        print(f"  ⚠  Could not fetch FIPS county table ({e})")
        return {}


# ── Census coordinate fallback (SECONDARY) ───────────────────────────────────

def census_reverse_single(lat: float, lon: float) -> dict:
    try:
        resp = requests.get(
            CENSUS_COORDS_URL,
            params={
                "x": lon, "y": lat,
                "benchmark": "Public_AR_Current",
                "vintage":   "Current_Current",
                "format":    "json",
            },
            timeout=10,
        )
        resp.raise_for_status()
        counties = (
            resp.json()
            .get("result", {})
            .get("geographies", {})
            .get("Counties", [])
        )
        if counties:
            c = counties[0]
            return {
                "county_name": c.get("NAME",  ""),
                "county_fips": c.get("GEOID", ""),
                "state_fips":  c.get("STATE", ""),
            }
    except Exception:
        pass
    return {}


def geocode_by_latlon(df: pd.DataFrame) -> dict:
    still_missing = (
        (df["county_fips"].isna() | (df["county_fips"] == "")) &
        df["latitude"].notna() & df["longitude"].notna()
    )
    target  = df[still_missing]
    indices = target.index.tolist()

    if not indices:
        print("  No remaining rows need lat/lon fallback.")
        return {}

    if DEBUG and DEBUG_ROWS:
        indices = indices[:DEBUG_ROWS]
        print(f"  ⚙  Debug mode: limiting lat/lon fallback to {DEBUG_ROWS} rows")

    print(f"  Lat/lon fallback for {len(indices):,} unmatched schools...")
    results = {}
    batches = [
        indices[i : i + COORD_BATCH_SIZE]
        for i in range(0, len(indices), COORD_BATCH_SIZE)
    ]

    for batch_num, batch_idx in enumerate(
        tqdm(batches, desc="  Coord batches", unit="batch"), start=1
    ):
        for idx in batch_idx:
            row = target.loc[idx]
            lat, lon = row.get("latitude"), row.get("longitude")
            if pd.notna(lat) and pd.notna(lon):
                result = census_reverse_single(float(lat), float(lon))
                if result:
                    results[idx] = result
                elif DEBUG and len(results) < 3:
                    print(f"  ⚠  No result for lat={lat}, lon={lon} "
                          f"(school: {row.get('name', '?')})")
            time.sleep(0.05)

        if batch_num < len(batches):
            time.sleep(CENSUS_RATE_DELAY)

    print(f"  ✓  Lat/lon fallback matched {len(results):,} / {len(indices):,} schools")
    return results


# ── County enrichment orchestrator ───────────────────────────────────────────

def enrich_counties(df: pd.DataFrame, use_address_batch: bool = True,
                    use_latlon: bool = True) -> pd.DataFrame:
    print("\n  ── County enrichment pipeline ──")

    # ── Step 1: seed from NCES ──
    df["county_name"] = df.get("county", pd.Series(dtype=str))
    df["county_fips"] = ""
    df["state_fips"]  = ""

    if DEBUG:
        # Show raw county values before any cleaning
        for stype in df["school_type"].unique():
            debug_nces_county(df[df["school_type"] == stype], stype)

    df["county_name"] = df["county_name"].apply(
        lambda v: "" if str(v).strip().lower() in NCES_BAD
        else str(v).strip() if pd.notna(v) else ""
    )

    from_nces    = (df["county_name"] != "").sum()
    still_needed = (df["county_name"] == "").sum()
    print(f"  Step 1 — NCES county field      : {from_nces:,} filled, "
          f"{still_needed:,} still missing")

    # ── Step 2: address batch ──
    if use_address_batch and still_needed > 0:
        print(f"\n  Step 2 — Census address batch geocoder")

        target_df = df[df["county_name"] == ""]

        if DEBUG:
            for stype in target_df["school_type"].unique():
                debug_address_sample(
                    target_df[target_df["school_type"] == stype], stype
                )

        print("  Fetching FIPS county name table...")
        fips_table = fetch_fips_county_table()
        print(f"  ✓  {len(fips_table):,} county FIPS codes loaded")

        # ── Step 1b: back-fill FIPS for rows that already have a county name ──
        # Private schools get county_name from NCES in Step 1 but never go
        # through the geocoder, so they have no county_fips. Build a reverse
        # lookup (lowercase name → fips) and fill those in now.
        name_to_fips = {v.lower(): k for k, v in fips_table.items()}
        has_name_no_fips = (df["county_name"] != "") & (df["county_fips"] == "")
        backfill_count = 0
        for idx in df[has_name_no_fips].index:
            cname = df.at[idx, "county_name"].lower()
            fips  = name_to_fips.get(cname, "")
            if not fips:
                # Try matching without " county" suffix
                fips = name_to_fips.get(cname.replace(" county", "").strip(), "")
            if fips:
                df.at[idx, "county_fips"] = fips
                df.at[idx, "state_fips"]  = fips[:2]
                backfill_count += 1
        print(f"  Step 1b — FIPS back-fill        : {backfill_count:,} private schools "
              f"matched county name → FIPS")

        addr_results = geocode_by_address(target_df)

        filled_addr = 0
        for idx, vals in addr_results.items():
            fips = vals.get("county_fips", "")
            df.at[idx, "county_fips"] = fips
            df.at[idx, "state_fips"]  = vals.get("state_fips", "")
            df.at[idx, "county_name"] = fips_table.get(fips, "")
            filled_addr += 1

        still_needed = (df["county_fips"] == "").sum()
        print(f"  Step 2 result                   : {filled_addr:,} filled via address, "
              f"{still_needed:,} still missing")
    else:
        fips_table = {}

    # ── Step 3: lat/lon fallback ──
    still_needed_latlon = (
        (df["county_fips"] == "") &
        df["latitude"].notna() & df["longitude"].notna()
    ).sum()

    if not use_latlon:
        print(f"\n  Step 3 — Lat/lon fallback       : skipped (--no-latlon)")
    elif still_needed_latlon > 0:
        print(f"\n  Step 3 — Census lat/lon fallback ({still_needed_latlon:,} schools)")
        coord_results = geocode_by_latlon(df)
        for idx, vals in coord_results.items():
            df.at[idx, "county_name"] = vals.get("county_name", "")
            df.at[idx, "county_fips"] = vals.get("county_fips", "")
            df.at[idx, "state_fips"]  = vals.get("state_fips",  "")
    else:
        print(f"\n  Step 3 — Lat/lon fallback       : not needed")

    total_filled = (df["county_name"] != "").sum()
    fips_filled  = (df["county_fips"] != "").sum()
    print(f"\n  ── Enrichment complete ──")
    print(f"  county_name filled : {total_filled:,} / {len(df):,} "
          f"({total_filled/len(df)*100:.1f}%)")
    print(f"  county_fips filled : {fips_filled:,} / {len(df):,} "
          f"({fips_filled/len(df)*100:.1f}%)")

    return df



# ── Smart title-case helper ───────────────────────────────────────────────────
# Handles edge cases like "McDonald", "McArthur", "O'Brien", short words
# (of, the, at, etc.), and all-caps abbreviations like "STEM".

_LOWERCASE_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "if", "in",
    "nor", "of", "on", "or", "so", "the", "to", "up", "yet",
}

def _smart_titlecase(s: str) -> str:
    """Convert a string to title case with common English rules."""
    if not s or not s.strip():
        return s
    words  = s.strip().split()
    result = []
    for i, word in enumerate(words):
        clean = word.strip(".,;:-()")
        # Preserve all-caps abbreviations of 2-3 chars (e.g. "IB", "STEM" handled below)
        if clean.isupper() and len(clean) <= 3 and clean.isalpha():
            result.append(word)
            continue
        # Always capitalize first and last word
        if i == 0 or i == len(words) - 1:
            result.append(word.capitalize())
            continue
        # Lowercase short connective words mid-sentence
        if clean.lower() in _LOWERCASE_WORDS:
            result.append(word.lower())
            continue
        result.append(word.capitalize())
    return " ".join(result)

# ── Main ──────────────────────────────────────────────────────────────────────

def build_database(do_geocode: bool = True, use_address_batch: bool = True,
                   use_latlon: bool = True):
    print("\n=== NCES School Database Builder ===\n")
    if DEBUG:
        print("  ⚙  DEBUG MODE ON — saving sent/received files to current directory\n")

    # ── Public schools ──
    print("1/5  Fetching public school data...")
    try:
        pub_df = clean(remap(first_csv(download_zip(PUBLIC_URL, "public schools")),
                             PUBLIC_COLS), "public")
        print(f"  ✓  {len(pub_df):,} public schools loaded")
    except Exception as e:
        print(f"  ✗  {e}\n     See https://nces.ed.gov/ccd/files.asp")
        pub_df = pd.DataFrame()

    # ── Private schools ──
    print("\n2/5  Fetching private school data...")
    try:
        prv_df = clean(remap(first_csv(download_zip(PRIVATE_URL, "private schools")),
                             PRIVATE_COLS), "private")
        print(f"  ✓  {len(prv_df):,} private schools loaded")
    except Exception as e:
        print(f"  ✗  {e}\n     See https://nces.ed.gov/surveys/pss/pssdata.asp")
        prv_df = pd.DataFrame()

    if pub_df.empty and prv_df.empty:
        print("\n✗  No data loaded. Exiting.")
        return

    # ── Combine ──
    print("\n3/5  Combining datasets...")
    combined = pd.concat([pub_df, prv_df], ignore_index=True)
    print(f"  ✓  {len(combined):,} total schools")

    # Populate diocese_name from lookup table for Catholic private schools
    if "diocese_code" in combined.columns:
        combined["diocese_name"] = combined["diocese_code"].apply(
            lambda v: DIOCESE_TABLE.get(str(v).strip().zfill(4), "")
            if pd.notna(v) and str(v).strip() not in ("", "0", "nan") else ""
        )
        matched = (combined["diocese_name"] != "").sum()
        print(f"  ✓  {matched:,} Catholic schools matched to diocese name")

    # ── County enrichment ──
    if do_geocode:
        print("\n4/5  Enriching county + FIPS data...")
        combined = enrich_counties(combined, use_address_batch=use_address_batch,
                                   use_latlon=use_latlon)
    else:
        print("\n4/5  Skipping geocoding (--no-geocode)")
        combined["county_name"] = combined.get("county", "")
        combined["county_fips"] = ""
        combined["state_fips"]  = ""


    # ── Title-case text fields ──
    print("  Formatting text fields...")
    # Columns to title-case (names, addresses, cities, etc.)
    titlecase_cols = [
        "name", "address", "city", "district_name", "diocese_name",
        "county_name", "lea_name",
    ]
    # Columns to uppercase (abbreviations, codes)
    upper_cols = ["state_abbr"]

    for col in titlecase_cols:
        if col in combined.columns:
            combined[col] = combined[col].apply(
                lambda v: _smart_titlecase(str(v)) if pd.notna(v) and str(v).strip() else v
            )
    for col in upper_cols:
        if col in combined.columns:
            combined[col] = combined[col].str.upper()

    # ── Reorder columns ──
    col_order = [
        "nces_id", "name", "school_type",
        "district_id", "district_name",
        "diocese_code", "diocese_name",
        "address", "city", "state_abbr", "zip",
        "county_name", "county_fips", "state_fips",
        "phone", "website", "latitude", "longitude",
        "grade_low", "grade_high", "enrollment",
        "title1_status", "locale_code",
    ]
    combined = combined[[c for c in col_order if c in combined.columns]]

    # ── Save ──
    print(f"\n5/5  Saving...")
    conn = sqlite3.connect(DB_PATH)
    combined.to_sql("schools", conn, if_exists="replace", index=False)
    for idx_sql in [
        "CREATE INDEX IF NOT EXISTS idx_state        ON schools(state_abbr)",
        "CREATE INDEX IF NOT EXISTS idx_type         ON schools(school_type)",
        "CREATE INDEX IF NOT EXISTS idx_zip          ON schools(zip)",
        "CREATE INDEX IF NOT EXISTS idx_name         ON schools(name)",
        "CREATE INDEX IF NOT EXISTS idx_county_fips  ON schools(county_fips)",
        "CREATE INDEX IF NOT EXISTS idx_district_id   ON schools(district_id)",
    ]:
        conn.execute(idx_sql)
    conn.commit()
    conn.close()
    print(f"  ✓  SQLite database → {DB_PATH}")

    combined.to_csv(CSV_PATH, index=False)
    print(f"  ✓  CSV export      → {CSV_PATH}")

    print("\n── Summary ──────────────────────────────────────")
    print(f"  Total schools  : {len(combined):,}")
    for stype, n in combined["school_type"].value_counts().items():
        print(f"  {stype.capitalize():10s}   : {n:,}")
    print(f"  States covered : {combined['state_abbr'].nunique()}")
    if "county_name" in combined.columns:
        n = (combined["county_name"].notna() & (combined["county_name"] != "")).sum()
        print(f"  With county    : {n:,} ({n/len(combined)*100:.1f}%)")
    if "county_fips" in combined.columns:
        n = (combined["county_fips"].notna() & (combined["county_fips"] != "")).sum()
        print(f"  With FIPS      : {n:,} ({n/len(combined)*100:.1f}%)")
    print("─────────────────────────────────────────────────\n")
    print("Example queries:")
    print("  sqlite3 schools.db \"SELECT * FROM schools WHERE state_abbr='OH' LIMIT 5;\"")
    print("  sqlite3 schools.db \"SELECT * FROM schools WHERE county_fips='39049';\"")
    print()


def query_examples():
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT state_abbr, school_type, COUNT(*) as count "
        "FROM schools GROUP BY state_abbr, school_type ORDER BY count DESC LIMIT 10",
        conn,
    )
    print("── Top 10 state/type combos ────────────────────")
    print(df.to_string(index=False))
    conn.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build NCES school database")
    parser.add_argument("--no-geocode",       action="store_true",
                        help="Skip all geocoding; use NCES county data as-is")
    parser.add_argument("--no-address-batch", action="store_true",
                        help="Skip address batch; use only lat/lon fallback")
    parser.add_argument("--no-latlon",        action="store_true",
                        help="Skip lat/lon fallback geocoding (faster, "
                             "relies on address batch only)")
    parser.add_argument("--debug",            action="store_true",
                        help="Enable debug mode: save sent/received files, "
                             "print diagnostics")
    parser.add_argument("--debug-rows",       type=int, default=None,
                        metavar="N",
                        help="In debug mode, only geocode first N rows "
                             "(e.g. --debug-rows 50 for a fast test)")
    args = parser.parse_args()

    DEBUG      = args.debug
    DEBUG_ROWS = args.debug_rows

    build_database(
        do_geocode        = not args.no_geocode,
        use_address_batch = not args.no_address_batch,
        use_latlon        = not args.no_latlon,
    )
    query_examples()