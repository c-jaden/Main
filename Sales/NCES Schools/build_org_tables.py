"""
District & Diocese Table Builder
==================================
Adds two new tables to the existing schools.db:

  districts  — one row per public school district (from NCES CCD LEA file)
  dioceses   — one row per Catholic diocese (aggregated from schools + DIOCESE_TABLE)

Usage:
    pip install requests pandas tqdm
    python build_org_tables.py

    # Print column names from the LEA file without building the DB:
    python build_org_tables.py --inspect-lea

Output:
    Adds 'districts' and 'dioceses' tables to schools.db
"""

import io
import os
import re
import sys
import zipfile
import sqlite3
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

# ── Configuration ─────────────────────────────────────────────────────────────

DB_PATH = "schools.db"

# NCES CCD LEA (district) file — same year as the school file
# Pattern: ccd_lea_029_YYYY_w_Na_MMDDYY.zip
# Update this URL when a newer version is released
# Source: https://nces.ed.gov/ccd/files.asp  (Nonfiscal > LEA > 2024-25)
LEA_URL = "https://nces.ed.gov/ccd/Data/zip/ccd_lea_029_2425_w_1a_073025.zip"

# ── Diocese lookup table (from NCES PSS 2021-22 User Manual) ─────────────────

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
    "5601": "Diocese of Cheyenne, WY",
}

# ── Title-case helper ─────────────────────────────────────────────────────────

_LOWERCASE_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "if", "in",
    "nor", "of", "on", "or", "so", "the", "to", "up", "yet",
}
_KEEP_UPPER = {
    "II", "III", "IV", "VI", "VII", "VIII", "IX",
    "PK", "TK", "KG", "STEM", "STEAM", "JROTC", "ROTC",
}

def _titlecase(s: str) -> str:
    if not s or not str(s).strip():
        return s
    words  = str(s).strip().split()
    result = []
    for i, word in enumerate(words):
        clean = word.strip(".,;:-()")
        if clean.upper() in _KEEP_UPPER:
            result.append(clean.upper())
            continue
        if i == 0 or i == len(words) - 1:
            result.append(word.capitalize())
            continue
        if clean.lower() in _LOWERCASE_WORDS:
            result.append(word.lower())
            continue
        result.append(word.capitalize())
    return " ".join(result)


# ── LEA column mapping ────────────────────────────────────────────────────────

LEA_COLS = {
    "LEAID":      "district_id",
    "LEA_NAME":   "name",
    "LSTREET1":   "address",
    "LCITY":      "city",
    "ST":         "state_abbr",
    "LZIP":       "zip",
    "PHONE":      "phone",
    "GSLO":       "grade_low",
    "GSHI":       "grade_high",
    "LEA_TYPE":   "lea_type_code",
    "FIPST":      "state_fips",
    "WEBSITE":    "website",
}


# ── Download helpers ──────────────────────────────────────────────────────────

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
    raise FileNotFoundError("No CSV found in zip")



# ── USCCB diocese address lookup (hardcoded from usccb.org/about/bishops-and-dioceses/all-dioceses)
# Source: USCCB website — addresses current as of 2025

USCCB_ADDRESSES = {
    'archdiocese of mobile': {"address": '400 Government Street', "city": 'Mobile', "state_abbr": 'AL', "zip": '36602', "website": 'https://mobarch.org/'},
    'diocese of birmingham': {"address": '2121 3rd Avenue North; P.O. Box 12047', "city": 'Birmingham', "state_abbr": 'AL', "zip": '35202-2047', "website": 'http://www.bhmdiocese.org/'},
    'archdiocese of anchorage-juneau': {"address": '225 Cordova Street', "city": 'Anchorage', "state_abbr": 'AK', "zip": '99501-2409', "website": 'http://www.aoaj.org'},
    'diocese of fairbanks': {"address": '1316 Peger Road', "city": 'Fairbanks', "state_abbr": 'AK', "zip": '99709-5199', "website": 'https://dioceseoffairbanks.org/'},
    'diocese of phoenix': {"address": '400 East Monroe Street', "city": 'Phoenix', "state_abbr": 'AZ', "zip": '85004-2336', "website": 'http://www.diocesephoenix.org/'},
    'diocese of tucson': {"address": 'P.O. Box 31', "city": 'Tucson', "state_abbr": 'AZ', "zip": '85702', "website": 'http://www.diocesetucson.org/'},
    'diocese of little rock': {"address": '2500 N. Tyler Street', "city": 'Little Rock', "state_abbr": 'AR', "zip": '72207', "website": 'http://www.dolr.org/'},
    'archdiocese of los angeles': {"address": '3424 Wilshire Boulevard', "city": 'Los Angeles', "state_abbr": 'CA', "zip": '90010-2202', "website": 'https://lacatholics.org/'},
    'archdiocese of san francisco': {"address": 'One Peter Yorke Way', "city": 'San Francisco', "state_abbr": 'CA', "zip": '94109', "website": 'http://www.sfarchdiocese.org/'},
    'diocese of fresno': {"address": '1550 North Fresno Street', "city": 'Fresno', "state_abbr": 'CA', "zip": '93707-3788', "website": 'https://dioceseoffresno.org/'},
    'diocese of monterey': {"address": '425 Church Street', "city": 'Monterey', "state_abbr": 'CA', "zip": '93940', "website": 'http://www.dioceseofmonterey.org/'},
    'diocese of oakland': {"address": '2121 Harrison Street Suite 100', "city": 'Oakland', "state_abbr": 'CA', "zip": '94612', "website": 'http://www.oakdiocese.org/'},
    'diocese of orange': {"address": '13280 Chapman Avenue', "city": 'Garden Grove', "state_abbr": 'CA', "zip": '92840', "website": 'http://www.rcbo.org/'},
    'diocese of sacramento': {"address": '2110 Broadway', "city": 'Sacramento', "state_abbr": 'CA', "zip": '95818', "website": 'https://www.scd.org/'},
    'diocese of san bernardino': {"address": '1201 E. Highland Avenue', "city": 'San Bernardino', "state_abbr": 'CA', "zip": '92404-5300', "website": 'http://www.sbdiocese.org/'},
    'diocese of san diego': {"address": 'P.O. Box 85728', "city": 'San Diego', "state_abbr": 'CA', "zip": '92186-5728', "website": 'https://sdcatholic.org/'},
    'diocese of san jose': {"address": '1150 North 1st Street Suite 100', "city": 'San Jose', "state_abbr": 'CA', "zip": '95112-4966', "website": 'http://www.dsj.org/'},
    'diocese of santa rosa': {"address": 'P.O. Box 1297', "city": 'Santa Rosa', "state_abbr": 'CA', "zip": '95402', "website": 'https://srdiocese.org/'},
    'diocese of stockton': {"address": '212 N. San Joaquin Street', "city": 'Stockton', "state_abbr": 'CA', "zip": '95202-2409', "website": 'http://www.stocktondiocese.org/'},
    'archdiocese of denver': {"address": '1300 South Steele Street', "city": 'Denver', "state_abbr": 'CO', "zip": '80210', "website": 'http://www.archden.org/'},
    'diocese of colorado springs': {"address": '228 N. Cascade Avenue', "city": 'Colorado Springs', "state_abbr": 'CO', "zip": '80903', "website": 'http://www.diocs.org/'},
    'diocese of pueblo': {"address": '101 N. Greenwood Street', "city": 'Pueblo', "state_abbr": 'CO', "zip": '81003-3164', "website": 'http://www.dioceseofpueblo.org/'},
    'archdiocese of hartford': {"address": '467 Bloomfield Avenue', "city": 'Bloomfield', "state_abbr": 'CT', "zip": '06002-2999', "website": 'http://www.archdioceseofhartford.org/'},
    'diocese of bridgeport': {"address": '100 Beard Sawmill Road Suite 650', "city": 'Shelton', "state_abbr": 'CT', "zip": '06848', "website": 'http://www.bridgeportdiocese.com/'},
    'diocese of norwich': {"address": '201 Broadway', "city": 'Norwich', "state_abbr": 'CT', "zip": '06360', "website": 'http://www.norwichdiocese.org/'},
    'diocese of wilmington': {"address": '1925 Delaware Avenue', "city": 'Wilmington', "state_abbr": 'DE', "zip": '19899', "website": 'http://www.cdow.org/'},
    'archdiocese of washington': {"address": '5001 Eastern Avenue', "city": 'Hyattsville', "state_abbr": 'MD', "zip": '20782', "website": 'http://www.adw.org/'},
    'archdiocese of miami': {"address": '9401 Biscayne Boulevard', "city": 'Miami Shores', "state_abbr": 'FL', "zip": '33138', "website": 'http://www.miamiarch.org/'},
    'diocese of pensacola-tallahassee': {"address": '11 North B Street', "city": 'Pensacola', "state_abbr": 'FL', "zip": '32502', "website": 'http://www.ptdiocese.org/'},
    'diocese of orlando': {"address": 'P.O. Box 1800', "city": 'Orlando', "state_abbr": 'FL', "zip": '32802-1800', "website": 'http://www.orlandodiocese.org/'},
    'diocese of palm beach': {"address": '9995 N. Military Trail', "city": 'Palm Beach Gardens', "state_abbr": 'FL', "zip": '33410', "website": 'http://www.diocesepb.org/'},
    'diocese of st. augustine': {"address": '11625 Old St. Augustine Rd.', "city": 'Jacksonville', "state_abbr": 'FL', "zip": '32258', "website": 'http://www.dosafl.com/'},
    'diocese of st. petersburg': {"address": '6363 9th Avenue N.', "city": 'St. Petersburg', "state_abbr": 'FL', "zip": '33710', "website": 'http://www.dosp.org/'},
    'diocese of venice': {"address": '1000 Pinebrook Road', "city": 'Venice', "state_abbr": 'FL', "zip": '34285', "website": 'http://www.dioceseofvenice.org/'},
    'archdiocese of atlanta': {"address": '2401 Lake Park Drive S.E.', "city": 'Smyrna', "state_abbr": 'GA', "zip": '30080', "website": 'http://www.archatl.com/'},
    'diocese of savannah': {"address": '2170 East Victory Drive', "city": 'Savannah', "state_abbr": 'GA', "zip": '31404-3918', "website": 'http://www.diosav.org/'},
    'diocese of honolulu': {"address": '1184 Bishop Street', "city": 'Honolulu', "state_abbr": 'HI', "zip": '96813', "website": 'http://www.catholichawaii.org/'},
    'diocese of boise': {"address": '1501 S. Federal Way Suite 400', "city": 'Boise', "state_abbr": 'ID', "zip": '83705', "website": 'http://www.catholicidaho.org/'},
    'archdiocese of chicago': {"address": '835 N. Rush Street', "city": 'Chicago', "state_abbr": 'IL', "zip": '60611-2030', "website": 'http://www.archchicago.org/'},
    'diocese of belleville': {"address": '222 South Third Street', "city": 'Belleville', "state_abbr": 'IL', "zip": '62220', "website": 'http://www.diobelle.org/'},
    'diocese of joliet': {"address": '16555 Weber Road', "city": 'Crest Hill', "state_abbr": 'IL', "zip": '60403', "website": 'http://www.dioceseofjoliet.org/'},
    'diocese of peoria': {"address": '419 N.E. Madison Avenue', "city": 'Peoria', "state_abbr": 'IL', "zip": '61603-3719', "website": 'http://www.cdop.org/'},
    'diocese of rockford': {"address": '555 Coleman Center Drive', "city": 'Rockford', "state_abbr": 'IL', "zip": '81108', "website": 'http://www.rockforddiocese.org/'},
    'diocese of springfield in illinois': {"address": '1615 West Washington Street', "city": 'Springfield', "state_abbr": 'IL', "zip": '62702-4757', "website": 'http://www.dio.org/'},
    'archdiocese of indianapolis': {"address": '1400 N. Meridian Street', "city": 'Indianapolis', "state_abbr": 'IN', "zip": '46202', "website": 'http://www.archindy.org/'},
    'diocese of evansville': {"address": '4200 N. Kentucky Avenue', "city": 'Evansville', "state_abbr": 'IN', "zip": '47724-0169', "website": 'https://www.evdio.org/'},
    'diocese of fort wayne-south bend': {"address": '915 South Clinton', "city": 'Fort Wayne', "state_abbr": 'IN', "zip": '46801', "website": 'http://www.diocesefwsb.org'},
    'diocese of gary': {"address": '9292 Broadway', "city": 'Merrillville', "state_abbr": 'IN', "zip": '46410', "website": 'http://www.dcgary.org/'},
    'diocese of lafayette in indiana': {"address": 'P.O. Box 260', "city": 'Lafayette', "state_abbr": 'IN', "zip": '47902-0260', "website": 'http://www.dol-in.org/'},
    'archdiocese of dubuque': {"address": '1229 Mt. Loretta Avenue', "city": 'Dubuque', "state_abbr": 'IA', "zip": '52003', "website": 'https://www.dbqarch.org/'},
    'diocese of davenport': {"address": '780 W. Central Park Av.', "city": 'Davenport', "state_abbr": 'IA', "zip": '52804-1901', "website": 'http://www.davenportdiocese.org/'},
    'diocese of des moines': {"address": '601 Grand Avenue', "city": 'Des Moines', "state_abbr": 'IA', "zip": '50309', "website": 'http://www.dmdiocese.org/'},
    'diocese of sioux city': {"address": '1821 Jackson Street', "city": 'Sioux City', "state_abbr": 'IA', "zip": '51102-3379', "website": 'http://www.scdiocese.org/'},
    'archdiocese of kansas city in kansas': {"address": '12615 Parallel Parkway', "city": 'Kansas City', "state_abbr": 'KS', "zip": '66109', "website": 'http://www.archkck.org/'},
    'diocese of dodge city': {"address": 'P.O. Box 137', "city": 'Dodge City', "state_abbr": 'KS', "zip": '67801', "website": 'http://www.dcdiocese.org/'},
    'diocese of salina': {"address": '103 North Ninth Street', "city": 'Salina', "state_abbr": 'KS', "zip": '67401-2503', "website": 'http://www.salinadiocese.org/'},
    'diocese of wichita': {"address": '424 N. Broadway', "city": 'Wichita', "state_abbr": 'KS', "zip": '67202', "website": 'http://www.catholicdioceseofwichita.org/'},
    'archdiocese of louisville': {"address": '3940 Poplar Level Road', "city": 'Louisville', "state_abbr": 'KY', "zip": '40213-1463', "website": 'http://www.archlou.org/'},
    'diocese of covington': {"address": '1125 Madison Avenue', "city": 'Covington', "state_abbr": 'KY', "zip": '41011-3115', "website": 'http://www.covingtondiocese.org/'},
    'diocese of lexington': {"address": '1310 W. Main Street', "city": 'Lexington', "state_abbr": 'KY', "zip": '40508-2048', "website": 'http://www.cdlex.org/'},
    'diocese of owensboro': {"address": '600 Locust Street', "city": 'Owensboro', "state_abbr": 'KY', "zip": '42301', "website": 'http://www.rcdok.org/'},
    'archdiocese of new orleans': {"address": '7887 Walmsley Avenue', "city": 'New Orleans', "state_abbr": 'LA', "zip": '70125', "website": 'http://www.arch-no.org/'},
    'diocese of alexandria': {"address": '4400 Coliseum Boulevard', "city": 'Alexandria', "state_abbr": 'LA', "zip": '71303-3513', "website": 'http://www.diocesealex.org/'},
    'diocese of baton rouge': {"address": 'P.O. Box 2028', "city": 'Baton Rouge', "state_abbr": 'LA', "zip": '70821-2028', "website": 'http://www.diobr.org/'},
    'diocese of houma-thibodaux': {"address": '2779 Highway 311', "city": 'Schriever', "state_abbr": 'LA', "zip": '70395', "website": 'http://www.htdiocese.org/'},
    'diocese of lafayette': {"address": '1408 Carmel Drive', "city": 'Lafayette', "state_abbr": 'LA', "zip": '70501-5215', "website": 'http://www.diolaf.org/'},
    'diocese of lake charles': {"address": '414 Iris Street', "city": 'Lake Charles', "state_abbr": 'LA', "zip": '70601-5234', "website": 'http://www.lcdiocese.org/'},
    'diocese of shreveport': {"address": '3500 Fairfield Avenue', "city": 'Shreveport', "state_abbr": 'LA', "zip": '71104', "website": 'http://www.dioshpt.org/'},
    'diocese of portland': {"address": '510 Ocean Avenue', "city": 'Portland', "state_abbr": 'ME', "zip": '04103-4936', "website": 'http://www.portlanddiocese.net/'},
    'archdiocese of baltimore': {"address": '320 Cathedral Street', "city": 'Baltimore', "state_abbr": 'MD', "zip": '21201', "website": 'http://www.archbalt.org/'},
    'archdiocese of boston': {"address": '66 Brooks Drive', "city": 'Braintree', "state_abbr": 'MA', "zip": '02184-3839', "website": 'http://www.bostoncatholic.org'},
    'diocese of fall river': {"address": '450 Highland Avenue', "city": 'Fall River', "state_abbr": 'MA', "zip": '02720', "website": 'http://www.fallriverdiocese.org/'},
    'diocese of springfield': {"address": 'P.O. Box 1730', "city": 'Springfield', "state_abbr": 'MA', "zip": '01102', "website": 'http://www.diospringfield.org/'},
    'diocese of worcester': {"address": '49 Elm Street', "city": 'Worcester', "state_abbr": 'MA', "zip": '01609', "website": 'http://www.worcesterdiocese.org/'},
    'archdiocese of detroit': {"address": '12 State Street', "city": 'Detroit', "state_abbr": 'MI', "zip": '48226', "website": 'http://www.aod.org/'},
    'diocese of grand rapids': {"address": '360 Division Avenue S.', "city": 'Grand Rapids', "state_abbr": 'MI', "zip": '49503-4501', "website": 'http://www.dioceseofgrandrapids.org/'},
    'diocese of gaylord': {"address": '611 North Street', "city": 'Gaylord', "state_abbr": 'MI', "zip": '49735', "website": 'http://www.dioceseofgaylord.org/'},
    'diocese of kalamazoo': {"address": '215 N. Westnedge Avenue', "city": 'Kalamazoo', "state_abbr": 'MI', "zip": '49007', "website": 'http://www.diokzoo.org/'},
    'diocese of lansing': {"address": '228 N. Walnut Street', "city": 'Lansing', "state_abbr": 'MI', "zip": '48933-1119', "website": 'http://www.dioceseoflansing.org/'},
    'diocese of marquette': {"address": '1004 Harbor Hills Drive', "city": 'Marquette', "state_abbr": 'MI', "zip": '49855', "website": 'http://www.dioceseofmarquette.org/'},
    'diocese of saginaw': {"address": '5800 Weiss Street', "city": 'Saginaw', "state_abbr": 'MI', "zip": '48603', "website": 'http://www.dioceseofsaginaw.org/'},
    'archdiocese of st. paul and minneapolis': {"address": '226 Summit Avenue', "city": 'St. Paul', "state_abbr": 'MN', "zip": '55102', "website": 'http://www.archspm.org/'},
    'diocese of crookston': {"address": '1200 Memorial Drive', "city": 'Crookston', "state_abbr": 'MN', "zip": '56716-2102', "website": 'http://www.crookston.org/'},
    'diocese of duluth': {"address": '2830 E. 4th Street', "city": 'Duluth', "state_abbr": 'MN', "zip": '55812', "website": 'http://www.dioceseduluth.org/'},
    'diocese of new ulm': {"address": '1400 6th Street North', "city": 'New Ulm', "state_abbr": 'MN', "zip": '56073-2099', "website": 'http://www.dnu.org/'},
    'diocese of st. cloud': {"address": 'P.O. Box 1248', "city": 'St. Cloud', "state_abbr": 'MN', "zip": '56302-1248', "website": 'http://www.clouddiocese.org/'},
    'diocese of winona-rochester': {"address": '55 W. Sanborn Street', "city": 'Winona', "state_abbr": 'MN', "zip": '55987', "website": 'http://www.dow.org/'},
    'diocese of winona': {"address": '55 W. Sanborn Street', "city": 'Winona', "state_abbr": 'MN', "zip": '55987', "website": 'http://www.dow.org/'},
    'diocese of biloxi': {"address": '1790 Popps Ferry Road', "city": 'Biloxi', "state_abbr": 'MS', "zip": '39532', "website": 'http://www.biloxidiocese.org/'},
    'diocese of jackson': {"address": 'P.O. Box 2248', "city": 'Jackson', "state_abbr": 'MS', "zip": '39225-2248', "website": 'http://www.jacksondiocese.org/'},
    'archdiocese of st. louis': {"address": '20 Archbishop May Drive', "city": 'St. Louis', "state_abbr": 'MO', "zip": '63119-5004', "website": 'http://www.archstl.org/'},
    'diocese of jefferson city': {"address": '2207 W. Main Street', "city": 'Jefferson City', "state_abbr": 'MO', "zip": '65109', "website": 'http://www.diojeffcity.org/'},
    'diocese of kansas city-st. joseph': {"address": '20 West 9th Street', "city": 'Kansas City', "state_abbr": 'MO', "zip": '64105', "website": 'http://www.diocese-kcsj.org/'},
    'diocese of springfield-cape girardeau': {"address": '601 S. Jefferson Avenue', "city": 'Springfield', "state_abbr": 'MO', "zip": '65806', "website": 'http://www.dioscg.org/'},
    'diocese of great falls-billings': {"address": 'P.O. Box 1399', "city": 'Great Falls', "state_abbr": 'MT', "zip": '59403', "website": 'http://www.diocesegfb.org/'},
    'diocese of helena': {"address": 'P.O. Box 1729', "city": 'Helena', "state_abbr": 'MT', "zip": '59624', "website": 'http://www.dioceseofhelena.org/'},
    'archdiocese of omaha': {"address": '2222 Castelar Street', "city": 'Omaha', "state_abbr": 'NE', "zip": '68103', "website": 'http://www.archomaha.org/'},
    'diocese of grand island': {"address": 'P.O. Box 996', "city": 'Grand Island', "state_abbr": 'NE', "zip": '68802', "website": 'http://www.gidiocese.org/'},
    'diocese of lincoln': {"address": '3400 Sheridan Boulevard', "city": 'Lincoln', "state_abbr": 'NE', "zip": '68506', "website": 'http://www.lincolndiocese.org/'},
    'diocese of las vegas': {"address": '302 Cathedral Way', "city": 'Las Vegas', "state_abbr": 'NV', "zip": '89109', "website": 'http://www.lvdiocese.org/'},
    'diocese of reno': {"address": '290 S. Arlington Avenue', "city": 'Reno', "state_abbr": 'NV', "zip": '89501', "website": 'http://www.dioceseofreno.org/'},
    'diocese of manchester': {"address": '153 Ash Street', "city": 'Manchester', "state_abbr": 'NH', "zip": '03104', "website": 'http://www.catholicnh.org/'},
    'archdiocese of newark': {"address": '171 Clifton Avenue', "city": 'Newark', "state_abbr": 'NJ', "zip": '07104', "website": 'http://www.rcan.org/'},
    'diocese of camden': {"address": '631 Market Street', "city": 'Camden', "state_abbr": 'NJ', "zip": '08102-1107', "website": 'http://www.camdendiocese.org/'},
    'diocese of metuchen': {"address": 'P.O. Box 191', "city": 'Metuchen', "state_abbr": 'NJ', "zip": '08840', "website": 'http://www.diometuchen.org/'},
    'diocese of paterson': {"address": '777 Valley Road', "city": 'Clifton', "state_abbr": 'NJ', "zip": '07013', "website": 'http://www.rcdop.org/'},
    'diocese of trenton': {"address": '701 Lawrenceville Road', "city": 'Trenton', "state_abbr": 'NJ', "zip": '08648-3418', "website": 'http://www.dioceseoftrenton.org/'},
    'archdiocese of santa fe': {"address": '4000 St. Joseph Place N.W.', "city": 'Albuquerque', "state_abbr": 'NM', "zip": '87120', "website": 'http://www.archdiosf.org/'},
    'diocese of gallup': {"address": 'P.O. Box 1338', "city": 'Gallup', "state_abbr": 'NM', "zip": '87305', "website": 'http://www.dioceseofgallup.org/'},
    'diocese of las cruces': {"address": '1280 Med Park Drive', "city": 'Las Cruces', "state_abbr": 'NM', "zip": '88005', "website": 'http://www.dioceseoflascruces.org/'},
    'archdiocese of new york': {"address": '1011 First Avenue', "city": 'New York', "state_abbr": 'NY', "zip": '10022', "website": 'http://www.archny.org/'},
    'diocese of albany': {"address": '40 North Main Avenue', "city": 'Albany', "state_abbr": 'NY', "zip": '12203', "website": 'http://www.rcda.org/'},
    'diocese of brooklyn': {"address": '310 Prospect Park West', "city": 'Brooklyn', "state_abbr": 'NY', "zip": '11215', "website": 'http://www.dioceseofbrooklyn.org/'},
    'diocese of buffalo': {"address": '795 Main Street', "city": 'Buffalo', "state_abbr": 'NY', "zip": '14203', "website": 'http://www.buffalodiocese.org/'},
    'diocese of ogdensburg': {"address": 'P.O. Box 369', "city": 'Ogdensburg', "state_abbr": 'NY', "zip": '13669', "website": 'http://www.rcdony.org/'},
    'diocese of rochester': {"address": 'P.O. Box 22397', "city": 'Rochester', "state_abbr": 'NY', "zip": '14692-2397', "website": 'http://www.dor.org/'},
    'diocese of rockville centre': {"address": 'P.O. Box 9023', "city": 'Rockville Centre', "state_abbr": 'NY', "zip": '11571-9023', "website": 'http://www.drvc.org/'},
    'diocese of syracuse': {"address": '240 E. Onondaga Street', "city": 'Syracuse', "state_abbr": 'NY', "zip": '13202', "website": 'http://www.syracusediocese.org/'},
    'diocese of charlotte': {"address": '1123 S. Church Street', "city": 'Charlotte', "state_abbr": 'NC', "zip": '28203-4003', "website": 'http://www.charlottediocese.org/'},
    'diocese of raleigh': {"address": '715 Nazareth Street', "city": 'Raleigh', "state_abbr": 'NC', "zip": '27606', "website": 'http://www.dioceseofraleigh.org/'},
    'diocese of bismarck': {"address": '520 N. Washington Street', "city": 'Bismarck', "state_abbr": 'ND', "zip": '58501', "website": 'http://www.bismarckdiocese.com/'},
    'diocese of fargo': {"address": '5201 Bishops Boulevard Suite A', "city": 'Fargo', "state_abbr": 'ND', "zip": '58104-7605', "website": 'http://www.fargodiocese.org/'},
    'archdiocese of cincinnati': {"address": '100 E. Eighth Street', "city": 'Cincinnati', "state_abbr": 'OH', "zip": '45202', "website": 'http://www.catholiccincinnati.org/'},
    'diocese of cleveland': {"address": '1404 E. Ninth Street', "city": 'Cleveland', "state_abbr": 'OH', "zip": '44114-1785', "website": 'http://www.dioceseofcleveland.org/'},
    'diocese of columbus': {"address": '198 E. Broad Street', "city": 'Columbus', "state_abbr": 'OH', "zip": '43215-3766', "website": 'http://www.columbuscatholic.org/'},
    'diocese of steubenville': {"address": '422 Washington Street', "city": 'Steubenville', "state_abbr": 'OH', "zip": '43952', "website": 'http://www.diosteub.org/'},
    'diocese of toledo': {"address": '1933 Spielbusch Avenue', "city": 'Toledo', "state_abbr": 'OH', "zip": '43604', "website": 'http://www.toledodiocese.org/'},
    'diocese of youngstown': {"address": '144 W. Wood Street', "city": 'Youngstown', "state_abbr": 'OH', "zip": '44503', "website": 'http://www.doy.org/'},
    'archdiocese of oklahoma city': {"address": 'P.O. Box 32180', "city": 'Oklahoma City', "state_abbr": 'OK', "zip": '73123', "website": 'http://www.archokc.org/'},
    'diocese of tulsa': {"address": 'P.O. Box 690240', "city": 'Tulsa', "state_abbr": 'OK', "zip": '74169-0240', "website": 'http://www.dioceseoftulsa.org/'},
    'archdiocese of portland in oregon': {"address": '2838 E. Burnside Street', "city": 'Portland', "state_abbr": 'OR', "zip": '97214', "website": 'http://www.archdpdx.org/'},
    'archdiocese of portland': {"address": '2838 E. Burnside Street', "city": 'Portland', "state_abbr": 'OR', "zip": '97214', "website": 'http://www.archdpdx.org/'},
    'diocese of baker': {"address": 'P.O. Box 5999', "city": 'Bend', "state_abbr": 'OR', "zip": '97708', "website": 'http://www.dioceseofbaker.org/'},
    'archdiocese of philadelphia': {"address": '222 N. 17th Street', "city": 'Philadelphia', "state_abbr": 'PA', "zip": '19103-1299', "website": 'http://www.archphila.org/'},
    'diocese of allentown': {"address": 'P.O. Box F', "city": 'Allentown', "state_abbr": 'PA', "zip": '18105', "website": 'http://www.allentowndiocese.org/'},
    'diocese of altoona-johnstown': {"address": 'P.O. Box 409', "city": 'Hollidaysburg', "state_abbr": 'PA', "zip": '16648', "website": 'http://www.dioceseaj.org/'},
    'diocese of erie': {"address": '429 E. Grandview Boulevard', "city": 'Erie', "state_abbr": 'PA', "zip": '16504-1960', "website": 'http://www.eriercd.org/'},
    'diocese of greensburg': {"address": '723 E. Pittsburgh Street', "city": 'Greensburg', "state_abbr": 'PA', "zip": '15601-3808', "website": 'http://www.dioceseofgreensburg.org/'},
    'diocese of harrisburg': {"address": '4800 Union Deposit Road', "city": 'Harrisburg', "state_abbr": 'PA', "zip": '17111-3710', "website": 'http://www.hbgdiocese.org/'},
    'diocese of pittsburgh': {"address": '111 Boulevard of the Allies', "city": 'Pittsburgh', "state_abbr": 'PA', "zip": '15222', "website": 'http://www.diopitt.org/'},
    'diocese of scranton': {"address": '300 Wyoming Avenue', "city": 'Scranton', "state_abbr": 'PA', "zip": '18503', "website": 'http://www.dioceseofscranton.org/'},
    'diocese of providence': {"address": 'One Cathedral Square', "city": 'Providence', "state_abbr": 'RI', "zip": '02903', "website": 'http://www.dioceseofprovidence.org/'},
    'diocese of charleston': {"address": '901 Orange Grove Road', "city": 'Charleston', "state_abbr": 'SC', "zip": '29407', "website": 'http://www.charlestondiocese.org/'},
    'diocese of rapid city': {"address": '606 Cathedral Drive', "city": 'Rapid City', "state_abbr": 'SD', "zip": '57701', "website": 'http://www.rapidcitydiocese.org/'},
    'diocese of sioux falls': {"address": '523 N. Duluth Avenue', "city": 'Sioux Falls', "state_abbr": 'SD', "zip": '57104-2714', "website": 'http://www.sfcatholic.org/'},
    'diocese of knoxville': {"address": '805 Northshore Drive S.W.', "city": 'Knoxville', "state_abbr": 'TN', "zip": '37919', "website": 'http://www.dioknox.org/'},
    'diocese of memphis': {"address": '5825 Shelby Oaks Drive', "city": 'Memphis', "state_abbr": 'TN', "zip": '38134', "website": 'http://www.memphiscatholic.org/'},
    'diocese of nashville': {"address": '2800 McGavock Pike', "city": 'Nashville', "state_abbr": 'TN', "zip": '37214', "website": 'http://www.dioceseofnashville.com/'},
    'archdiocese of san antonio': {"address": '2718 W. Woodlawn Avenue', "city": 'San Antonio', "state_abbr": 'TX', "zip": '78228-5116', "website": 'http://www.archsa.org/'},
    'diocese of amarillo': {"address": 'P.O. Box 5644', "city": 'Amarillo', "state_abbr": 'TX', "zip": '79117-5644', "website": 'http://www.amarillodiocese.org/'},
    'diocese of austin': {"address": 'P.O. Box 13327', "city": 'Austin', "state_abbr": 'TX', "zip": '78711', "website": 'http://www.austindiocese.org/'},
    'diocese of beaumont': {"address": 'P.O. Box 3948', "city": 'Beaumont', "state_abbr": 'TX', "zip": '77704', "website": 'http://www.dioceseofbmt.org/'},
    'diocese of brownsville': {"address": 'P.O. Box 2279', "city": 'Brownsville', "state_abbr": 'TX', "zip": '78522', "website": 'http://www.cdob.org/'},
    'diocese of corpus christi': {"address": '620 Lipan Street', "city": 'Corpus Christi', "state_abbr": 'TX', "zip": '78401-2519', "website": 'http://www.diocesecc.org/'},
    'diocese of dallas': {"address": '3725 Blackburn Street', "city": 'Dallas', "state_abbr": 'TX', "zip": '75219', "website": 'http://www.cathdal.org/'},
    'diocese of el paso': {"address": '499 St. Matthews Street', "city": 'El Paso', "state_abbr": 'TX', "zip": '79907', "website": 'http://www.elpasodiocese.org/'},
    'diocese of fort worth': {"address": '800 W. Loop 820 South', "city": 'Fort Worth', "state_abbr": 'TX', "zip": '76108-2919', "website": 'http://www.fwdioc.org/'},
    'diocese of galveston-houston': {"address": '1700 San Jacinto Street', "city": 'Houston', "state_abbr": 'TX', "zip": '77002-8291', "website": 'http://www.archgh.org/'},
    'diocese of lubbock': {"address": 'P.O. Box 98700', "city": 'Lubbock', "state_abbr": 'TX', "zip": '79499', "website": 'http://www.catholiclubbock.org/'},
    'diocese of san angelo': {"address": 'P.O. Box 1829', "city": 'San Angelo', "state_abbr": 'TX', "zip": '76902', "website": 'http://www.sanangelodiocese.org/'},
    'diocese of tyler': {"address": '1015 ESE Loop 323', "city": 'Tyler', "state_abbr": 'TX', "zip": '75701-9663', "website": 'http://www.dioceseoftyler.org/'},
    'diocese of victoria': {"address": 'P.O. Box 4070', "city": 'Victoria', "state_abbr": 'TX', "zip": '77903', "website": 'http://www.victoriadiocese.org/'},
    'diocese of laredo': {"address": '1901 Corpus Christi Street', "city": 'Laredo', "state_abbr": 'TX', "zip": '78043', "website": 'http://www.dioceseoflaredo.org/'},
    'diocese of salt lake city': {"address": '27 C Street', "city": 'Salt Lake City', "state_abbr": 'UT', "zip": '84103', "website": 'http://www.dioslc.org/'},
    'diocese of salt lake': {"address": '27 C Street', "city": 'Salt Lake City', "state_abbr": 'UT', "zip": '84103', "website": 'http://www.dioslc.org/'},
    'diocese of burlington': {"address": '55 Joy Drive', "city": 'South Burlington', "state_abbr": 'VT', "zip": '05403', "website": 'http://www.vermontcatholic.org/'},
    'diocese of arlington': {"address": '200 N. Glebe Road', "city": 'Arlington', "state_abbr": 'VA', "zip": '22203', "website": 'http://www.arlingtondiocese.org/'},
    'diocese of richmond': {"address": '7800 Carousel Lane', "city": 'Richmond', "state_abbr": 'VA', "zip": '23294-4201', "website": 'http://www.richmonddiocese.org/'},
    'archdiocese of seattle': {"address": '710 9th Avenue', "city": 'Seattle', "state_abbr": 'WA', "zip": '98104', "website": 'http://www.seattlearch.org/'},
    'diocese of spokane': {"address": 'P.O. Box 1453', "city": 'Spokane', "state_abbr": 'WA', "zip": '99210-1453', "website": 'http://www.dioceseofspokane.org/'},
    'diocese of yakima': {"address": '5301-A Tieton Drive', "city": 'Yakima', "state_abbr": 'WA', "zip": '98908', "website": 'http://www.yakimadiocese.org/'},
    'diocese of wheeling-charleston': {"address": '1300 Byron Street', "city": 'Wheeling', "state_abbr": 'WV', "zip": '26003', "website": 'http://www.dwc.org/'},
    'archdiocese of milwaukee': {"address": 'P.O. Box 070912', "city": 'Milwaukee', "state_abbr": 'WI', "zip": '53207-0912', "website": 'http://www.archmil.org/'},
    'diocese of green bay': {"address": 'P.O. Box 23825', "city": 'Green Bay', "state_abbr": 'WI', "zip": '54305-3825', "website": 'http://www.gbdioc.org/'},
    'diocese of la crosse': {"address": '3710 East Avenue South', "city": 'La Crosse', "state_abbr": 'WI', "zip": '54602-4004', "website": 'http://www.diolc.org/'},
    'diocese of madison': {"address": '702 S. High Point Road', "city": 'Madison', "state_abbr": 'WI', "zip": '53719-3735', "website": 'http://www.madisondiocese.org/'},
    'diocese of superior': {"address": '1201 Hughitt Avenue', "city": 'Superior', "state_abbr": 'WI', "zip": '54880', "website": 'http://www.catholicdioceseofsuperiror.org/'},
    'diocese of cheyenne': {"address": '2121 Capitol Avenue', "city": 'Cheyenne', "state_abbr": 'WY', "zip": '82001', "website": 'http://www.dioceseofcheyenne.org/'},
}


def _lookup_diocese_address(diocese_name: str) -> dict:
    """Match a DIOCESE_TABLE name to USCCB address data."""
    if not diocese_name:
        return {}
    # Strip state suffix from NCES name e.g. "Diocese of Columbus, OH" -> "Diocese of Columbus"
    norm = re.sub(r",\s*[A-Z]{2}$", "", diocese_name).strip().lower()
    if norm in USCCB_ADDRESSES:
        return USCCB_ADDRESSES[norm]
    # Try partial match
    for key, val in USCCB_ADDRESSES.items():
        if norm in key or key in norm:
            return val
    return {}


# ── Districts table ───────────────────────────────────────────────────────────

def build_districts(conn: sqlite3.Connection):
    print("\n1/2  Building districts table...")

    try:
        files  = download_zip(LEA_URL, "NCES LEA district file")
        raw    = first_csv(files)
    except Exception as e:
        print(f"  ✗  Could not download LEA file: {e}")
        print(f"     URL tried: {LEA_URL}")
        print("     Update LEA_URL at the top of this script with the correct URL")
        print("     from https://nces.ed.gov/ccd/files.asp (Nonfiscal > LEA > 2024-25)")
        return

    # Print all columns for inspection
    available = {k: v for k, v in LEA_COLS.items() if k in raw.columns}
    missing   = [k for k in LEA_COLS if k not in raw.columns]
    if missing:
        print(f"  ⚠  Columns not found (may vary by year): {missing}")

    df = raw[list(available.keys())].rename(columns=available)

    # Clean up NCES sentinel values
    bad = {"N", "M", "-1", "-2", "†", "‡", "–", "nan"}
    for col in df.columns:
        df[col] = df[col].apply(
            lambda v: "" if str(v).strip() in bad else str(v).strip()
            if pd.notna(v) else ""
        )

    # Title-case text fields
    for col in ["name", "address", "city"]:
        if col in df.columns:
            df[col] = df[col].apply(_titlecase)

    # Add school count and enrollment from existing schools table
    print("  Joining school counts from schools table...")
    school_stats = pd.read_sql(
        """
        SELECT district_id,
               COUNT(*)         AS school_count,
               SUM(CAST(enrollment AS INTEGER)) AS total_enrollment
        FROM   schools
        WHERE  school_type = 'public'
          AND  district_id IS NOT NULL
          AND  district_id != ''
        GROUP  BY district_id
        """,
        conn,
    )
    df = df.merge(school_stats, on="district_id", how="left")
    df["school_count"]      = df["school_count"].fillna(0).astype(int)
    df["total_enrollment"]  = df["total_enrollment"].fillna(0).astype(int)

    # Save to DB
    df.to_sql("districts", conn, if_exists="replace", index=False)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dist_id    ON districts(district_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dist_state ON districts(state_abbr)"
    )
    conn.commit()

    print(f"  ✓  {len(df):,} districts saved to 'districts' table")


# ── Dioceses table ────────────────────────────────────────────────────────────

def build_dioceses(conn: sqlite3.Connection):
    print("\n2/2  Building dioceses table...")

    # Aggregate from private schools already in DB
    try:
        school_data = pd.read_sql(
            """
            SELECT diocese_code,
                   state_abbr,
                   COUNT(*)                           AS school_count,
                   SUM(CAST(enrollment AS INTEGER))   AS total_enrollment
            FROM   schools
            WHERE  school_type = 'private'
              AND  diocese_code IS NOT NULL
              AND  diocese_code != ''
              AND  diocese_code != '0'
            GROUP  BY diocese_code, state_abbr
            """,
            conn,
        )
    except Exception as e:
        print(f"  ✗  Could not read schools table: {e}")
        return

    if school_data.empty:
        print("  ⚠  No private schools with diocese_code found in schools table.")
        print("     Make sure build_school_database.py has been run first.")
        return

    # Roll up to one row per diocese_code
    agg = (
        school_data
        .groupby("diocese_code")
        .agg(
            school_count    = ("school_count",     "sum"),
            total_enrollment= ("total_enrollment", "sum"),
            states          = ("state_abbr",        lambda x: ", ".join(sorted(x.dropna().unique()))),
        )
        .reset_index()
    )

    # Add diocese name from lookup table
    agg["diocese_code"] = agg["diocese_code"].apply(
        lambda v: str(v).strip().zfill(4)
    )
    agg["name"] = agg["diocese_code"].map(DIOCESE_TABLE).fillna("")

    # Extract state from diocese name (last 2 chars after comma+space)
    agg["primary_state"] = agg["name"].apply(
        lambda v: v.split(", ")[-1] if ", " in v else ""
    )

    # Look up addresses from hardcoded USCCB table
    # Drop state_abbr if it exists from the groupby aggregation to avoid conflicts
    agg = agg.drop(columns=["state_abbr"], errors="ignore")
    agg["address"]    = agg["name"].apply(lambda n: _lookup_diocese_address(n).get("address", ""))
    agg["city"]       = agg["name"].apply(lambda n: _lookup_diocese_address(n).get("city", ""))
    agg["state_abbr"] = agg["name"].apply(lambda n: _lookup_diocese_address(n).get("state_abbr", ""))
    agg["zip"]        = agg["name"].apply(lambda n: _lookup_diocese_address(n).get("zip", ""))
    agg["phone"]      = ""  # Not available in USCCB directory; populate manually or via diocese websites
    agg["dio_website"]= agg["name"].apply(lambda n: _lookup_diocese_address(n).get("website", ""))

    matched_addr = (agg["address"] != "").sum()
    print(f"  ✓  {matched_addr:,} / {len(agg):,} dioceses matched to USCCB address")

    # Reorder columns
    col_order = [
        "diocese_code", "name", "primary_state", "states",
        "address", "city", "state_abbr", "zip", "phone", "dio_website",
        "school_count", "total_enrollment",
    ]
    agg = agg[[c for c in col_order if c in agg.columns]]

    agg.to_sql("dioceses", conn, if_exists="replace", index=False)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dioc_code  ON dioceses(diocese_code)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dioc_state ON dioceses(primary_state)"
    )
    conn.commit()

    print(f"  ✓  {len(agg):,} dioceses saved to 'dioceses' table")


# ── Inspect helper ────────────────────────────────────────────────────────────

def inspect_lea():
    """Download the LEA file and print all column names — useful for debugging."""
    print("Inspecting LEA file columns...")
    try:
        files = download_zip(LEA_URL, "NCES LEA file")
        raw   = first_csv(files)
        print(f"\nAll columns ({len(raw.columns)}):")
        for col in sorted(raw.columns):
            print(f"  {col}")
        print(f"\nSample row:")
        print(raw.head(1).T.to_string())
    except Exception as e:
        print(f"  ✗  {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build districts and dioceses tables in schools.db"
    )
    parser.add_argument(
        "--inspect-lea", action="store_true",
        help="Print LEA file column names and exit (useful for debugging)"
    )
    args = parser.parse_args()

    if args.inspect_lea:
        inspect_lea()
        return

    if not os.path.exists(DB_PATH):
        print(f"✗  {DB_PATH} not found. Run build_school_database.py first.")
        sys.exit(1)

    print("\n=== District & Diocese Table Builder ===\n")
    conn = sqlite3.connect(DB_PATH)

    build_districts(conn)
    build_dioceses(conn)

    # ── Export combined CSV ──
    print("\n3/3  Exporting combined CSV...")
    try:
        districts_df = pd.read_sql("SELECT *, 'district' AS org_type FROM districts", conn)
        dioceses_df  = pd.read_sql("SELECT *, 'diocese'  AS org_type FROM dioceses",  conn)

        # Combine — columns that don't exist in one table will be NaN
        combined = pd.concat([districts_df, dioceses_df], ignore_index=True)

        # Put org_type first for clarity
        cols = ["org_type"] + [c for c in combined.columns if c != "org_type"]
        combined = combined[cols]

        csv_path = r"C:\Users\jat27\Documents\Hope Squad\Main\HS Data\inbox\districts_dioceses.csv"
        combined.to_csv(csv_path, index=False)
        print(f"  ✓  Combined CSV saved → {csv_path}")
        print(f"     Districts : {len(districts_df):,} rows")
        print(f"     Dioceses  : {len(dioceses_df):,} rows")
        print(f"     Total     : {len(combined):,} rows")
    except Exception as e:
        print(f"  ✗  Could not export CSV: {e}")

    # ── Summary ──
    print("\n── Summary ──────────────────────────────────────")
    for table in ("districts", "dioceses"):
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:12s} : {n:,} rows")
        except Exception:
            print(f"  {table:12s} : not created")
    print("─────────────────────────────────────────────────\n")
    print("Example queries:")
    print("  -- All districts in Ohio")
    print("  SELECT * FROM districts WHERE state_abbr = 'OH';")
    print()
    print("  -- Schools per district, sorted by size")
    print("  SELECT name, city, school_count, total_enrollment")
    print("  FROM districts ORDER BY school_count DESC LIMIT 20;")
    print()
    print("  -- All Catholic schools in a diocese")
    print("  SELECT s.name, s.city, s.state_abbr")
    print("  FROM schools s JOIN dioceses d ON s.diocese_code = d.diocese_code")
    print("  WHERE d.name = 'Diocese of Columbus, OH';")

    conn.close()


if __name__ == "__main__":
    main()