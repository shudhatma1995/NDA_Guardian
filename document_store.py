"""
document_store.py — Pre-parsed NDA clause storage and retrieval.

No PDF parser needed: document is assumed to arrive as plain text.
Clause segmentation uses regex on numbered/titled sections.
"""

import re

# Global clause store (populated by load_document or load_sample)
CLAUSES: dict[str, str] = {}

# Canonical clause name mappings (normalized aliases → canonical key)
_CLAUSE_ALIASES: dict[str, str] = {
    "confidentiality": "confidentiality",
    "confidential": "confidentiality",
    "nda": "confidentiality",
    "non_compete": "non_compete",
    "non-compete": "non_compete",
    "noncompete": "non_compete",
    "non compete": "non_compete",
    "restrictive covenant": "non_compete",
    "ip_assignment": "ip_assignment",
    "ip assignment": "ip_assignment",
    "intellectual property": "ip_assignment",
    "ip": "ip_assignment",
    "work product": "ip_assignment",
    "indemnification": "indemnification",
    "indemnity": "indemnification",
    "liability_cap": "liability_cap",
    "liability cap": "liability_cap",
    "limitation of liability": "liability_cap",
    "liability": "liability_cap",
    "term": "term",
    "term and termination": "term",
    "termination": "term",
    "duration": "term",
    "governing_law": "governing_law",
    "governing law": "governing_law",
    "jurisdiction": "governing_law",
    "dispute resolution": "governing_law",
}

# Section header patterns → canonical clause key
_SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)confidential", "confidentiality"),
    (r"(?i)non.?compete|restrictive covenant", "non_compete"),
    (r"(?i)intellectual property|ip assignment|work product", "ip_assignment"),
    (r"(?i)indemnif", "indemnification"),
    (r"(?i)limitation of liability|liability cap", "liability_cap"),
    (r"(?i)^term\b|term and termination|termination", "term"),
    (r"(?i)governing law|dispute resolution|jurisdiction", "governing_law"),
    (r"(?i)general provisions|miscellaneous", "general"),
]


def _identify_clause(header_text: str) -> str:
    """Map a section header string to a canonical clause key."""
    for pattern, key in _SECTION_PATTERNS:
        if re.search(pattern, header_text):
            return key
    # Fallback: slugify the header
    return re.sub(r"[^a-z0-9]+", "_", header_text.lower().strip()).strip("_")


def load_document(text: str) -> dict[str, str]:
    """
    Parse plain-text NDA into clause segments.

    Splits on HR lines (---) to isolate numbered sections.
    The preamble (before the first section) is stored as 'parties'.
    Numbered sections ("1. TITLE") are mapped to canonical clause keys.

    Returns {canonical_clause_key: clause_text} and updates global CLAUSES.
    """
    global CLAUSES
    CLAUSES = {}

    # Split on HR lines (---) first to get rough sections
    hr_pattern = re.compile(r"\n\s*[-=]{3,}\s*\n")
    rough_sections = hr_pattern.split(text)

    for i, section in enumerate(rough_sections):
        section = section.strip()
        if not section:
            continue

        # Numbered section pattern: "1. TITLE" or "2.1 Sub-title" at line start
        numbered_match = re.match(
            r"^(\d+(?:\.\d+)*)\.\s+([A-Z][A-Z\s,()&/'-]{3,})",
            section,
        )

        if numbered_match:
            # Use the title text (not the preamble title) to identify clause
            header = numbered_match.group(2).strip()
            key = _identify_clause(header)
            if key in CLAUSES:
                CLAUSES[key] += "\n\n" + section
            else:
                CLAUSES[key] = section
        else:
            # Non-numbered section — treat as preamble/parties if it mentions them
            if re.search(r"(?i)(between|party|parties|corporation|individual|agreement)", section):
                existing = CLAUSES.get("parties", "")
                CLAUSES["parties"] = (existing + "\n\n" + section).strip()

    return CLAUSES


def load_sample() -> dict[str, str]:
    """Load the bundled sample NDA from demo/sample_nda.txt."""
    import os
    sample_path = os.path.join(os.path.dirname(__file__), "demo", "sample_nda.txt")
    with open(sample_path, "r", encoding="utf-8") as f:
        text = f.read()
    return load_document(text)


def get_clause(clause_name: str) -> str | None:
    """
    Return clause text by name (tries aliases and canonical keys).
    Returns None if not found.
    """
    key = _normalize_clause_name(clause_name)
    return CLAUSES.get(key)


def _normalize_clause_name(clause_name: str) -> str:
    """Normalize a clause name string to a canonical key."""
    lowered = clause_name.lower().strip()
    return _CLAUSE_ALIASES.get(lowered, lowered.replace("-", "_").replace(" ", "_"))


def get_clause_summary(clause_name: str, max_words: int = 80) -> str:
    """
    Return a privacy-safe summary of a clause (first max_words words).

    - Strips numbered section headers (e.g. "2. NON-COMPETE COVENANT")
    - Strips actual party names (replaces with 'Party A'/'Party B')
    - Truncates at a sentence boundary
    - Returns empty string if clause not found
    """
    clause_text = get_clause(clause_name)
    if not clause_text:
        return ""

    # Strip section header lines (e.g. "2. NON-COMPETE COVENANT" or "3.1 Sub-heading")
    lines = clause_text.split("\n")
    content_lines = [
        line for line in lines
        if not re.match(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z]", line)
        and not re.match(r"^\s*[A-Z][A-Z\s,()&/'-]{6,}\s*$", line)  # ALL-CAPS title lines
    ]
    body_text = "\n".join(content_lines).strip()
    if not body_text:
        body_text = clause_text  # fallback to original

    # Privacy: replace real party names
    sanitized = _strip_party_names(body_text)

    # Truncate to max_words at a sentence boundary
    words = sanitized.split()
    if len(words) <= max_words:
        return sanitized.strip()

    # Find last sentence boundary within max_words
    truncated = " ".join(words[:max_words])
    last_sentence_end = max(
        truncated.rfind(". "),
        truncated.rfind("! "),
        truncated.rfind("? "),
    )
    if last_sentence_end > len(truncated) // 4:
        return truncated[: last_sentence_end + 1].strip()
    return truncated.strip() + "..."


def _strip_party_names(text: str) -> str:
    """Replace known party names with generic placeholders."""
    # Extract party names from CLAUSES["parties"] if available
    party_a = "Party A"
    party_b = "Party B"

    parties_text = CLAUSES.get("parties", "")
    # Look for company name pattern
    company_match = re.search(r"([A-Z][a-z]+ (?:Corp|Inc|LLC|Ltd)[.,])", parties_text)
    if company_match:
        company_name = company_match.group(1).rstrip(".,")
        text = text.replace(company_name, party_a)

    # Look for individual name pattern (Firstname Lastname)
    individual_match = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+),? an individual", parties_text)
    if individual_match:
        individual_name = individual_match.group(1)
        text = text.replace(individual_name, party_b)

    # Also generalize role labels
    text = re.sub(r"\b(the Company)\b", party_a, text)
    text = re.sub(r"\b(the Employee)\b", party_b, text)

    return text


def extract_parties() -> dict[str, str]:
    """
    Extract the legal names of parties from the loaded document.
    Returns {"company": "...", "individual": "..."} or populated fields.
    """
    parties_text = CLAUSES.get("parties", "")
    full_text = "\n".join(CLAUSES.values())

    result = {}

    # Company pattern
    company_match = re.search(
        r"([A-Z][a-zA-Z\s]+ (?:Corp|Inc|LLC|Ltd|Corporation|Limited))[,.]?",
        full_text,
    )
    if company_match:
        result["company"] = company_match.group(1).strip()

    # Individual pattern
    individual_match = re.search(
        r"([A-Z][a-z]+ [A-Z][a-z]+),? an individual",
        full_text,
    )
    if individual_match:
        result["individual"] = individual_match.group(1).strip()

    return result


def get_field_from_clause(clause_name: str, field: str) -> str:
    """
    Extract a specific field value from a clause.

    Supported fields: duration, scope, amount, parties, definition
    """
    clause_text = get_clause(clause_name)
    if not clause_text:
        return f"Clause '{clause_name}' not found in document."

    field = field.lower().strip()

    if field == "duration":
        # Look for time period patterns (handles "24 months", "twenty-four (24) months", etc.)
        patterns = [
            # Word form followed by optional parenthetical and unit: "twenty-four (24) months"
            r"(twenty-four|twenty four|twelve|six|thirty-six|thirty six|two|three)\s*(?:\(\d+\)\s*)?(months?|years?|days?)",
            # Numeric form: "24 months"
            r"(\d+)\s*(months?|years?|days?)",
        ]
        for pat in patterns:
            m = re.search(pat, clause_text, re.IGNORECASE)
            if m:
                return m.group(0).strip()
        return "Duration not explicitly stated."

    elif field == "scope":
        # Look for geographic or subject matter scope
        geo_match = re.search(r"(\d+)\s*mile\s*radius[^.]+\.", clause_text, re.IGNORECASE)
        if geo_match:
            return geo_match.group(0).strip()
        state_match = re.search(r"(?:State of|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", clause_text)
        if state_match:
            return f"Geographic scope: {state_match.group(0).strip()}"
        return "Scope details not found."

    elif field == "amount":
        # Look for dollar amounts
        amount_match = re.search(r"\$[\d,]+(?:\.\d{2})?(?:\s*USD)?", clause_text)
        if amount_match:
            return amount_match.group(0).strip()
        return "No monetary amount found."

    elif field == "parties":
        parties = extract_parties()
        if parties:
            return "; ".join(f"{k.title()}: {v}" for k, v in parties.items())
        return "Parties not found."

    elif field == "definition":
        # Return first substantive sentence of the clause
        sentences = re.split(r"(?<=[.!?])\s+", clause_text.strip())
        for s in sentences:
            if len(s.split()) > 10:
                return s.strip()
        return clause_text[:300].strip()

    else:
        # Default: return first 150 chars of clause
        return clause_text[:300].strip()
