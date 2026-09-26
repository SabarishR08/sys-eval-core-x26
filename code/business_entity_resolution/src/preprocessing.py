"""
High-Precision Text Normalization Engine for Synthetic Entity Resolution.
Inverts artificial noise: word-order shuffling, abbreviation swaps, US/Indian state expansions,
and extracts order-invariant numeric and token signatures.
"""
import re
from typing import Dict, List, Set, Tuple

# US State abbreviations to canonical full names
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

LEGAL_SUFFIXES = {
    # US / UK / International
    "corp": "corporation", "corporation": "corporation",
    "inc": "incorporated", "incorporated": "incorporated",
    "ltd": "limited", "limited": "limited",
    "pvt": "private", "private": "private",
    "llc": "llc", "llp": "llp",
    "co": "company", "company": "company",
    "gmbh": "gmbh", "plc": "plc",
    "ent": "enterprise", "enterprise": "enterprise", "enterprises": "enterprise",
    "grp": "group", "group": "group", "trust": "trust",
    "holdings": "holdings", "services": "services", "solutions": "solutions",
    "technologies": "technologies", "tech": "technologies",
    # France
    "sa": "sa", "sarl": "sarl", "sas": "sas", "sasu": "sasu",
    "sci": "sci", "snc": "snc", "eurl": "eurl",
    "cie": "compagnie", "compagnie": "compagnie",
    "ets": "etablissements", "societe": "societe",
}

ADDRESS_ABBREVIATIONS = {
    # Roadways & Units
    "rd": "road", "road": "road",
    "st": "street", "street": "street",
    "ave": "avenue", "avenue": "avenue",
    "blvd": "boulevard", "boulevard": "boulevard",
    "dr": "drive", "drive": "drive",
    "ln": "lane", "lane": "lane",
    "ct": "court", "court": "court",
    "pl": "place", "place": "place",
    "ste": "suite", "suite": "suite",
    "apt": "apartment", "apartment": "apartment",
    "bldg": "building", "building": "building",
    "fl": "floor", "floor": "floor",
    "hwy": "highway", "highway": "highway",
    "opp": "opposite", "opposite": "opposite",
    "nr": "near", "near": "near",
    "pk": "park", "pkwy": "parkway",
    "sq": "square", "ctr": "center", "cir": "circle",
    # France
    "bd": "boulevard", "bvd": "boulevard", "av": "avenue",
    "r": "rue", "rue": "rue", "all": "allee", "imp": "impasse",
    "chem": "chemin", "rte": "route", "bat": "batiment",
}


def clean_text(text: str) -> str:
    """Basic lowercasing, punctuation stripping, symbol normalization."""
    if not text or not isinstance(text, str):
        return ""
    text = text.replace("&", " and ")
    # Replace non-alphanumeric except hyphen and slash (used in municipal Indian addresses e.g. 8-2-67/1/A)
    text = re.sub(r"[^\w\s\-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def extract_numbers_and_pins(text: str) -> Set[str]:
    """
    Extracts precise numeric address tokens, PIN codes, door numbers, and complex municipal IDs.
    e.g. '8-2-67/1/A/3/1', '129', '35', '560100'
    """
    if not text or not isinstance(text, str):
        return set()
    cleaned = text.lower()
    # Complex address numbers like 8-2-67/1/a/3/1 or standard numbers
    tokens = re.findall(r"\b\d+[\d\-/\w]*\b", cleaned)
    valid_nums = set()
    for t in tokens:
        # Ignore ordinal indicators like '2nd', '3rd', '1st' if isolated
        if re.match(r"^\d+(st|nd|rd|th)$", t):
            continue
        # Only retain tokens containing at least one digit
        if any(c.isdigit() for c in t):
            valid_nums.add(t)
    return valid_nums


def normalize_business_name(name: str) -> Tuple[str, frozenset]:
    """
    Normalizes business name and returns both standard text and order-invariant token set.
    """
    cleaned = clean_text(name)
    tokens = cleaned.split()
    normalized_tokens = []
    for t in tokens:
        t_norm = LEGAL_SUFFIXES.get(t, t)
        # Filter purely decorative noise words
        if t_norm not in ["the", "of", "and", "in", "at"]:
            normalized_tokens.append(t_norm)

    norm_str = " ".join(normalized_tokens)
    return norm_str, frozenset(normalized_tokens)


def normalize_address(address: str, country: str = "") -> Tuple[str, frozenset, Set[str]]:
    """
    Normalizes address, expands roadway abbreviations and US states,
    and returns (normalized_str, token_set, numeric_tokens).
    """
    cleaned = clean_text(address)
    tokens = cleaned.split()
    normalized_tokens = []

    is_us = "us" in str(country).lower()

    for t in tokens:
        if is_us and t in US_STATES:
            # Expand US state abbreviation e.g. NC -> north carolina
            normalized_tokens.extend(US_STATES[t].split())
        else:
            t_norm = ADDRESS_ABBREVIATIONS.get(t, t)
            normalized_tokens.append(t_norm)

    norm_str = " ".join(normalized_tokens)
    num_tokens = extract_numbers_and_pins(address)
    return norm_str, frozenset(normalized_tokens), num_tokens
