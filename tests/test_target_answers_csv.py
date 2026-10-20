"""The target-answers CSV artifact never carries a live spreadsheet formula (OWASP CSV injection rule)."""

import csv
import io

import pytest

from aisc_plugin_interface.system_under_test import _as_csv


def _read(data: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(data.decode())))


@pytest.mark.parametrize("cell", ["=1+1", "+1+1", "-1+1", "@SUM(A1)", "\tx", "\rx", '=HYPERLINK("http://x")'])
def test_a_cell_that_a_spreadsheet_would_run_is_prefixed_with_a_quote(cell):
    rows = _read(_as_csv([{"prompt": cell, "answer": "ok"}]))
    assert rows[1] == ["'" + cell, "ok"]


def test_a_column_name_that_a_spreadsheet_would_run_is_prefixed_too():
    rows = _read(_as_csv([{"=cmd": "a"}]))
    assert rows[0] == ["'=cmd"]
    assert rows[1] == ["a"]


def test_ordinary_text_numbers_and_empty_cells_are_left_as_they_are():
    rows = _read(_as_csv([{"prompt": "hello = world", "score": -3, "ratio": 0.5, "note": ""}]))
    assert rows[0] == ["prompt", "score", "ratio", "note"]
    assert rows[1] == ["hello = world", "-3", "0.5", ""]


def test_a_row_missing_a_column_still_writes_an_empty_cell():
    rows = _read(_as_csv([{"a": "=x"}, {"b": "y"}]))
    assert rows[1:] == [["'=x", ""], ["", "y"]]
