"""Phase 6 — HTML Table Parsing with BeautifulSoup."""

from src.phase6_parse.parser import (
    SKIP_PATTERNS,
    THAI_DIGIT_MAP,
    extract_vote_cell,
    has_consistent_column,
    parse_html_table,
    parse_votes_from_markdown,
)

__all__ = [
    "SKIP_PATTERNS",
    "THAI_DIGIT_MAP",
    "extract_vote_cell",
    "has_consistent_column",
    "parse_html_table",
    "parse_votes_from_markdown",
]
