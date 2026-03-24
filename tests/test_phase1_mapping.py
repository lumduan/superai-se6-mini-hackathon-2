"""Tests for Phase 1 — Data Inventory & ID Mapping."""

import pytest
import pandas as pd

from src.phase1_mapping.mapping import (
    parse_id,
    build_document_groups,
    find_image_pages,
    load_submission_template,
    build_inventory,
    DocumentGroup,
    SubmissionRow,
)
from src.config import SUBMISSION_TEMPLATE, IMAGES_DIR


# ── parse_id ───────────────────────────────────────────────────────────────

class TestParseId:
    def test_constituency(self):
        assert parse_id("constituency_10_1_3") == ("constituency", 10, 1, 3)

    def test_party_list(self):
        assert parse_id("party_list_31_4_57") == ("party_list", 31, 4, 57)

    def test_large_numbers(self):
        id_type, prov, dist, row = parse_id("constituency_100_200_999")
        assert (id_type, prov, dist, row) == ("constituency", 100, 200, 999)

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Unexpected ID format"):
            parse_id("bad_id_format")

    def test_missing_row_raises(self):
        with pytest.raises(ValueError):
            parse_id("constituency_10_1")


# ── build_document_groups ─────────────────────────────────────────────────

class TestBuildDocumentGroups:
    def _make_df(self, ids: list[str]) -> pd.DataFrame:
        from src.phase1_mapping.mapping import parse_id
        rows = []
        for id_ in ids:
            id_type, prov, dist, row_num = parse_id(id_)
            rows.append({"id": id_, "votes": 0, "id_type": id_type,
                         "province": prov, "district": dist, "row_number": row_num})
        return pd.DataFrame(rows)

    def test_groups_by_doc_key(self):
        ids = ["constituency_10_1_1", "constituency_10_1_2", "constituency_10_2_1"]
        df = self._make_df(ids)
        groups = build_document_groups(df)
        assert "constituency_10_1" in groups
        assert "constituency_10_2" in groups
        assert groups["constituency_10_1"].expected_row_count == 2

    def test_constituency_and_party_list_separate(self):
        ids = ["constituency_10_1_1", "party_list_10_1_1"]
        df = self._make_df(ids)
        groups = build_document_groups(df)
        assert "constituency_10_1" in groups
        assert "party_list_10_1" in groups

    def test_rows_sorted_by_row_number(self):
        ids = ["constituency_10_1_3", "constituency_10_1_1", "constituency_10_1_2"]
        df = self._make_df(ids)
        groups = build_document_groups(df)
        row_numbers = [r.row_number for r in groups["constituency_10_1"].rows]
        assert row_numbers == sorted(row_numbers)

    def test_ids_list(self):
        ids = ["constituency_10_1_1", "constituency_10_1_2"]
        df = self._make_df(ids)
        groups = build_document_groups(df)
        assert groups["constituency_10_1"].ids == ids


# ── find_image_pages ──────────────────────────────────────────────────────

class TestFindImagePages:
    def test_real_images_exist(self, tmp_path):
        # Create fake images for a test doc
        (tmp_path / "constituency_10_1.png").write_bytes(b"fake")
        (tmp_path / "constituency_10_1_page2.png").write_bytes(b"fake")
        found = find_image_pages("constituency_10_1", tmp_path)
        assert len(found) == 2

    def test_party_list_uses_constituency_images(self, tmp_path):
        # party_list shares constituency scan files
        (tmp_path / "constituency_10_1.png").write_bytes(b"fake")
        found = find_image_pages("party_list_10_1", tmp_path)
        assert len(found) == 1

    def test_missing_images_returns_empty(self, tmp_path):
        found = find_image_pages("constituency_99_99", tmp_path)
        assert found == []


# ── Integration against real data ─────────────────────────────────────────

class TestBuildInventoryIntegration:
    """Smoke tests against the actual submission template and images."""

    @pytest.fixture(scope="class")
    def inventory(self):
        return build_inventory(SUBMISSION_TEMPLATE, IMAGES_DIR)

    def test_total_rows(self, inventory):
        groups, _ = inventory
        total = sum(g.expected_row_count for g in groups.values())
        assert total == 10053

    def test_all_doc_keys_are_strings(self, inventory):
        groups, _ = inventory
        assert all(isinstance(k, str) for k in groups)

    def test_no_empty_groups(self, inventory):
        groups, _ = inventory
        assert all(g.expected_row_count > 0 for g in groups.values())

    def test_pages_dict_covers_all_docs(self, inventory):
        groups, pages = inventory
        assert set(groups.keys()) == set(pages.keys())

    def test_constituency_doc_count(self, inventory):
        groups, _ = inventory
        c_count = sum(1 for g in groups.values() if g.id_type == "constituency")
        assert c_count > 0

    def test_party_list_doc_count(self, inventory):
        groups, _ = inventory
        p_count = sum(1 for g in groups.values() if g.id_type == "party_list")
        assert p_count > 0
