"""Debug a single document — useful during OCR tuning."""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import IMAGES_DIR, SUBMISSION_TEMPLATE
from src.phase1_mapping.mapping import build_inventory, print_inventory_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug a single document through the pipeline")
    parser.add_argument("doc_key", help="e.g. constituency_10_1")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    groups, pages = build_inventory(SUBMISSION_TEMPLATE, IMAGES_DIR)

    if args.doc_key not in groups:
        print(f"ERROR: {args.doc_key!r} not found in template.")
        print("Available keys (sample):", list(groups.keys())[:5])
        return

    grp = groups[args.doc_key]
    pg = pages[args.doc_key]
    print(f"\nDocument : {args.doc_key}")
    print(f"Rows     : {grp.expected_row_count}")
    print(f"Pages    : {len(pg)}")
    for p in pg:
        print(f"  {p}")
    print(f"\nIDs ({len(grp.ids)}):")
    for id_ in grp.ids:
        print(f"  {id_}")


if __name__ == "__main__":
    main()
