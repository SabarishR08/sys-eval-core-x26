"""
Text preprocessing and normalization routines for business entity resolution.
Handles legal suffix normalization, punctuation, abbreviations, and token extraction.
"""
import re
from typing import Dict, List, Set

LEGAL_SUFFIXES = {
    "corp": "corporation",
    "corporation": "corporation",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "ltd": "limited",
    "limited": "limited",
    "pvt": "private",
    "private": "private",
    "llc": "llc",
    "llp": "llp",
    "co": "company",
    "company": "company",
    "gmbh": "gmbh",
    "sarl": "sarl",
    "sa": "sa",
    "plc": "plc",
    "ent": "enterprise",
    "enterprise": "enterprise",
    "enterprises": "enterprise",
    "grp": "group",
    "group": "group",
}

ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "ste": "suite",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "hwy": "highway",
    "opp": "opposite",
    "nr": "near",
    "w/": "with",
    "pk": "park",
    "pkwy": "parkway",
}


def clean_text(text: str) -> str:
    """Basic lowercasing, symbol normalization, and whitespace cleanup."""
    if not text or not isinstance(text, str):
        return ""
    # Replace ampersands with and
    text = text.replace("&", " and ")
    # Replace punctuation with space
    text = re.sub(r"[^\w\s]", " ", text)
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def normalize_business_name(name: str) -> str:
    """Normalizes business name by expanding known legal suffixes and abbreviations."""
    cleaned = clean_text(name)
    tokens = cleaned.split()
    normalized_tokens = [LEGAL_SUFFIXES.get(t, t) for t in tokens]
    return " ".join(normalized_tokens)


def normalize_address(address: str) -> str:
    """Normalizes address by expanding standard roadway/location abbreviations."""
    cleaned = clean_text(address)
    tokens = cleaned.split()
    normalized_tokens = [ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(normalized_tokens)


def extract_numbers_and_pins(text: str) -> Set[str]:
    """Extracts numeric tokens (such as postal/ZIP codes, street numbers)."""
    if not text:
        return set()
    nums = re.findall(r"\b\d{2,10}\b", text)
    return set(nums)


def get_token_shingles(text: str, n: int = 3) -> Set[str]:
    """Generates character n-grams from normalized text for robust sub-word matching."""
    text = f" {text} "
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}
