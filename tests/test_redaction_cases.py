"""The connection client keeps its own copy of the platform's scrub (`_secrets_of` and `_scrub_all`: a
connection's key and secret header values, in the forms a system may echo them, become ***), since the interface
does not depend on aisc_redact. It runs the "echo" cases of the shared vectors, redaction-cases.json (in
shared/identity of the AISC checkout), and gives the shared rule's answers.

A checkout of the interface on its own has no vectors, and the test says so as a skip."""
import json
import os
from pathlib import Path

import pytest

from aisc_plugin_interface.connections import Descriptor, _scrub_all


def _cases_file():
    named = os.environ.get("AISC_REDACTION_CASES")
    if named:
        return Path(named)
    in_checkout = Path(__file__).resolve().parents[2] / "identity/redaction-cases.json"
    return in_checkout if in_checkout.is_file() else None


CASES_FILE = _cases_file()
CASES = json.loads(CASES_FILE.read_text())["echo"] if CASES_FILE else []


@pytest.mark.skipif(CASES_FILE is None, reason="no redaction-cases.json: the interface is not inside an AISC checkout")
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_the_client_takes_out_each_echoed_form_as_the_shared_vectors_say(case):
    d = Descriptor(name="c", label="c", kind="rest", base_url="http://target.example",
                   secret=(case["secrets"] or [None])[0])
    assert _scrub_all(case["in"], d) == case["out"]
