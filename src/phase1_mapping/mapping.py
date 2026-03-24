"""Phase 1 — Data Inventory & ID Mapping.

Builds a complete mapping from submission IDs to their source document
and expected row index.  No I/O side-effects — pure data transformation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import IMAGES_DIR, SUBMISSION_TEMPLATE

logger = logging.getLogger(__name__)

# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class SubmissionRow:
    id: str
    id_type: str          # 'constituency' or 'party_list'
    province: int
    district: int
    row_number: int
    doc_key: str


@dataclass
class DocumentGroup:
    doc_key: str
    id_type: str          # 'constituency' or 'party_list'
    province: int
    district: int
    rows: list[SubmissionRow] = field(default_factory=list)

    @property
    def expected_row_count(self) -> int:
        return len(self.rows)

    @property
    def ids(self) -> list[str]:
        return [r.id for r in self.rows]


# ── Core functions ─────────────────────────────────────────────────────────

# Matches both 'constituency_P_D_R' and 'party_list_P_D_R'
ID_PATTERN = re.compile(r"(constituency|party_list)_(\d+)_(\d+)_(\d+)")


def parse_id(id_str: str) -> tuple[str, int, int, int]:
    """Parse an ID string → (id_type, province, district, row).

    Supports:
      constituency_{province}_{district}_{row}
      party_list_{province}_{district}_{row}
    """
    m = ID_PATTERN.match(id_str)
    if not m:
        raise ValueError(f"Unexpected ID format: {id_str!r}")
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))


def load_submission_template(path: str | Path = SUBMISSION_TEMPLATE) -> pd.DataFrame:
    """Load template CSV and attach parsed columns."""
    df = pd.read_csv(path)
    parsed = df["id"].apply(
        lambda x: pd.Series(parse_id(x), index=["id_type", "province", "district", "row_number"])
    )
    df = pd.concat([df, parsed], axis=1)
    logger.info("Loaded template: %d rows across %d unique IDs", len(df), df["id"].nunique())
    return df


def build_document_groups(df: pd.DataFrame) -> dict[str, DocumentGroup]:
    """Group template rows by (id_type, province, district) → doc_key."""
    groups: dict[str, DocumentGroup] = {}

    for _, row in df.iterrows():
        id_type = str(row["id_type"])
        prov = int(row["province"])
        dist = int(row["district"])
        # Use id_type prefix so constituency and party_list docs stay separate
        doc_key = f"{id_type}_{prov}_{dist}"

        if doc_key not in groups:
            groups[doc_key] = DocumentGroup(
                doc_key=doc_key, id_type=id_type, province=prov, district=dist
            )

        groups[doc_key].rows.append(
            SubmissionRow(
                id=row["id"],
                id_type=id_type,
                province=prov,
                district=dist,
                row_number=int(row["row_number"]),
                doc_key=doc_key,
            )
        )

    # Sort rows within each group by row_number to guarantee order
    for grp in groups.values():
        grp.rows.sort(key=lambda r: r.row_number)

    logger.info("Built %d document groups", len(groups))
    return groups


def find_image_pages(doc_key: str, images_dir: str | Path = IMAGES_DIR) -> list[Path]:
    """Return existing page paths for a document, in page order.

    Image filenames use 'constituency_' prefix only — party_list docs share
    the same scans, so we strip the id_type prefix and look for constituency images.
    """
    base = Path(images_dir)
    # Normalise: party_list_P_D → constituency_P_D (same physical scan)
    image_key = re.sub(r"^party_list_", "constituency_", doc_key)
    suffixes = ["", "_page2", "_page3", "_page4"]
    pages = [base / f"{image_key}{s}.png" for s in suffixes]
    found = [p for p in pages if p.exists()]
    logger.debug("%s: %d page(s) found (image_key=%s)", doc_key, len(found), image_key)
    return found


def build_inventory(
    template_path: str | Path = SUBMISSION_TEMPLATE,
    images_dir: str | Path = IMAGES_DIR,
) -> tuple[dict[str, DocumentGroup], dict[str, list[Path]]]:
    """Full Phase 1 entry point.

    Returns
    -------
    groups : mapping doc_key → DocumentGroup
    pages  : mapping doc_key → list of image paths
    """
    df = load_submission_template(template_path)
    groups = build_document_groups(df)

    pages: dict[str, list[Path]] = {}
    missing_images = 0
    for doc_key in groups:
        found = find_image_pages(doc_key, images_dir)
        pages[doc_key] = found
        if not found:
            missing_images += 1
            logger.warning("No images found for %s", doc_key)

    logger.info(
        "Inventory complete: %d docs, %d missing image sets",
        len(groups),
        missing_images,
    )
    return groups, pages


# ── CLI helper ─────────────────────────────────────────────────────────────

def print_inventory_summary(
    groups: dict[str, DocumentGroup],
    pages: dict[str, list[Path]],
) -> None:
    print(f"\n{'doc_key':<30} {'rows':>6} {'pages':>6}")
    print("-" * 46)
    for doc_key, grp in sorted(groups.items()):
        n_pages = len(pages.get(doc_key, []))
        print(f"{doc_key:<30} {grp.expected_row_count:>6} {n_pages:>6}")
    print("-" * 46)
    total_rows = sum(g.expected_row_count for g in groups.values())
    total_pages = sum(len(v) for v in pages.values())
    print(f"{'TOTAL':<30} {total_rows:>6} {total_pages:>6}\n")
