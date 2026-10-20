"""_through, the row loop of @dataset_through_target, with a fake client (no platform, no network)."""
import json

from aisc_plugin_interface.connections import Answer, EndpointError
from aisc_plugin_interface.system_under_test import _through


class Fake:
    """A client that sends each row as render_request does (json.dumps) and answers `reply(row)`."""

    def __init__(self, reply):
        self.reply, self.sent = reply, []

    def ask(self, row):
        self.sent.append(json.loads(json.dumps(row)))
        out = self.reply(row)
        if isinstance(out, Exception):
            raise out
        return out


ROWS = [{"amount": 1}, {"amount": 2}]


# A system's own fields named like the wrapper's columns

def test_an_error_field_in_a_structured_answer_does_not_stop_the_run():
    merged = _through(Fake(lambda r: Answer(text={"score": 1, "error": None})), ROWS, 1, 0)
    assert [m["target.score"] for m in merged] == [1, 1]
    assert all("target.error" not in m and m["target.answer.error"] is None for m in merged)


def test_a_system_field_cannot_overwrite_the_wrappers_refused_flag():
    merged = _through(Fake(lambda r: Answer(text={"refused": "x", "answer": "y", "refusal_reason": "z"})), ROWS, 1, 0)
    assert merged[0]["target.refused"] is False
    assert merged[0]["target.answer.refused"] == "x" and merged[0]["target.answer.answer"] == "y"
    assert merged[0]["target.answer.refusal_reason"] == "z"


def test_every_call_failing_still_stops_the_run():
    import pytest

    with pytest.raises(RuntimeError, match="none of the 2 rows: boom"):
        _through(Fake(lambda r: EndpointError("boom")), ROWS, 1, 0)


def test_one_failed_call_is_a_row_error():
    merged = _through(Fake(lambda r: EndpointError("boom") if r["amount"] == 2 else Answer(text="ok")), ROWS, 1, 0)
    assert merged[0]["target.answer"] == "ok" and merged[1]["target.error"] == "boom"


# Dates and decimals in a dataset are sent as JSON values

def test_a_dataframe_with_a_timestamp_column_is_sent_as_iso_text():
    import pytest

    pd = pytest.importorskip("pandas")
    table = pd.DataFrame({"when": pd.to_datetime(["2024-01-01", None]), "n": [1, 2]})
    fake = Fake(lambda r: Answer(text="ok"))
    merged = _through(fake, table, 1, 0)
    assert fake.sent == [{"when": "2024-01-01T00:00:00", "n": 1}, {"when": None, "n": 2}]
    assert [m["target.answer"] for m in merged] == ["ok", "ok"]


def test_decimal_date_and_datetime_cells_are_sent_as_json_values():
    from datetime import date, datetime
    from decimal import Decimal

    fake = Fake(lambda r: Answer(text="ok"))
    _through(fake, [{"d": Decimal("1.5"), "day": date(2024, 1, 2), "at": datetime(2024, 1, 2, 3, 4, 5)}], 1, 0)
    assert fake.sent == [{"d": 1.5, "day": "2024-01-02", "at": "2024-01-02T03:04:05"}]
