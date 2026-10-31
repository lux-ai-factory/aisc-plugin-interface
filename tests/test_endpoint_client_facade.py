"""Under a run pass, EndpointClient never holds the target's key: it resolves the connection without it,
takes a run key from the platform and asks the platform's ``aisc/ask``, which calls the system and
retries. The answer and the errors are the ones a direct call gives. Without a pass (a hand-made
descriptor, or the service token of older setups) the client calls the system directly, as before."""
import hashlib
import json
import re
from pathlib import Path

import pytest

from aisc_plugin_interface import connections as c
from aisc_plugin_interface.connections import (
    Answer, BlockedAddress, Descriptor, EndpointAuthError, EndpointBadResponse, EndpointClient, EndpointError,
    EndpointNotFound, EndpointTimeout, TargetHasNoEndpoint,
)

from conftest import FakePlugin

PID = "0f7c1e2a-aaaa-bbbb-cccc-1234567890ab"
KEY = "component:0b9c7a1e-0000-4000-8000-000000000002"
SCORER = {"name": "mcas-score", "label": "MCAS scorer", "kind": "rest", "method": "POST", "path": "/score",
          "headers": {}, "secret_header": "X-Auth-Token: {{secret}}", "body_template": "{{input}}",
          "response_path": "$", "refusal": {"status": [502], "path": "detail.reason"}, "model": None,
          "timeout_s": 5, "secret": "s3cr3t", "updated_at": "2026-10-04T10:00:00Z",
          "target": {"key": KEY, "kind": "component", "label": "Scoring engine"}}
WITHOUT_SECRET = {k: v for k, v in SCORER.items() if k != "secret"}
RESOLVE = f"/internal/projects/{PID}/targets/{KEY}/connection"
RUN_KEYS = f"/internal/projects/{PID}/targets/{KEY}/run-keys"
ASK_PATH = f"/internal/facade/{PID}/mcas-score/aisc/ask"
RUN_KEY = "aisc-run-abc"
ROW = {"amount": 2500, "months": 12}
OUTPUT = {"decision": "approve", "risk": {"p": 0.5}}
OK = {"output": OUTPUT, "refused": False, "refusal_reason": None, "status": 200, "latency_ms": 12}


def resolved(stub, **over):
    """The platform's resolve answer under a pass: no secret, has_secret instead."""
    return dict(WITHOUT_SECRET, has_secret=True, base_url=stub.base, allowed_hosts=[stub.host], denied_addresses=[],
                **over)


def issued(stub, ask_url=None, capabilities=("text", "history"), **over):
    answer = {"key": RUN_KEY, "expires_at": "2026-10-26T00:00:00+00:00", "connection": "mcas-score",
              "capabilities": list(capabilities), "target": SCORER["target"],
              "endpoints": {"aisc": {"ask_url": ask_url or stub.base + ASK_PATH}}}
    answer.update(over)
    return answer


@pytest.fixture
def facade(stub, monkeypatch):
    stub.route(RESOLVE, (200, resolved(stub)))
    stub.route(RUN_KEYS, (201, issued(stub)))
    stub.route(ASK_PATH, (200, OK))
    stub.route("/score", (200, OUTPUT))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_RUN_PASS", "pass-123")
    monkeypatch.delenv("PLATFORM_CONNECTIONS_TOKEN", raising=False)
    return stub


class CountingPlugin(FakePlugin):
    def __init__(self, inputs=None):
        super().__init__(inputs)
        self.uploads = []

    def upload_artifact(self, name, content):
        self.uploads.append(name)
        super().upload_artifact(name, content)


def target_plugin():
    return CountingPlugin({"target": {"value": f"target:{PID}/{KEY}"}})


def client(**kwargs):
    return EndpointClient.for_target(target_plugin(), **kwargs)


def asks(stub):
    return [r for r in stub.seen if r["path"] == ASK_PATH]


def failing(stub, status, body):
    """The exception ``ask`` raises when the platform's aisc/ask answers (status, body)."""
    stub.route(ASK_PATH, (status, body))
    cl = client(sleep=lambda s: None)
    with pytest.raises(EndpointError) as raised:
        cl.ask(ROW)
    return raised.value


# resolving under a pass

def test_under_a_run_pass_the_client_holds_no_secret(facade):
    cl = client()
    assert cl.descriptor.secret is None
    assert cl.run_key.ask_url == facade.base + ASK_PATH
    assert cl.run_key.capabilities == ("text", "history")
    assert RUN_KEY not in repr(cl.run_key)
    assert [(r["method"], r["path"]) for r in facade.seen] == [("GET", RESOLVE), ("POST", RUN_KEYS)]
    assert all(r["headers"]["x-aisc-run-pass"] == "pass-123" for r in facade.seen)


def test_a_secret_the_platform_still_returns_is_dropped(facade):
    facade.route(RESOLVE, (200, resolved(facade, secret="s3cr3t")))
    cl = client()
    assert cl.descriptor.secret is None
    cl.ask(ROW)
    assert [r for r in facade.seen if r["path"] == "/score"] == []
    for r in facade.seen:
        assert "s3cr3t" not in json.dumps(r["headers"]) and b"s3cr3t" not in r["body"]


# asking under a pass

def test_ask_posts_input_history_and_params_with_the_run_key(facade):
    client().ask({"amount": 2500, "months": 12})
    (req,) = asks(facade)
    assert req["method"] == "POST"
    assert req["headers"]["authorization"] == f"Bearer {RUN_KEY}"
    assert req["headers"]["content-type"] == "application/json"
    assert "x-aisc-run-pass" not in req["headers"]
    assert json.loads(req["body"]) == {"input": {"amount": 2500, "months": 12}, "history": [], "params": {}}


def test_the_answer_round_trips(facade):
    assert client().ask(ROW) == Answer(text=OUTPUT, refused=False, refusal_reason=None, status=200, latency_ms=12,
                                       tool_calls=None)


def test_a_refusal_round_trips(facade):
    facade.route(ASK_PATH, (200, {"output": None, "refused": True, "refusal_reason": "outside the markets",
                                  "status": 502, "latency_ms": 5}))
    answer = client().ask(ROW)
    assert answer.refused is True
    assert answer.refusal_reason == "outside the markets"
    assert answer.text is None
    assert answer.status == 502


def test_text_history_and_params_round_trip(facade):
    facade.route(ASK_PATH, (200, dict(OK, output="No.")))
    hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    answer = client().ask("Is nationality an input?", history=hist, lang="en")
    (req,) = asks(facade)
    assert json.loads(req["body"]) == {"input": "Is nationality an input?", "history": hist, "params": {"lang": "en"}}
    assert answer.text == "No."


FACADE_CODES = [
    ("auth", 401, EndpointAuthError),
    ("not_found", 404, EndpointNotFound),
    ("timeout", None, EndpointTimeout),
    ("bad_response", 422, EndpointBadResponse),
    ("blocked", None, BlockedAddress),
    ("error", None, EndpointError),
    ("weird", None, EndpointError),
]


@pytest.mark.parametrize("code, status, error", FACADE_CODES)
def test_each_facade_error_code_is_the_exception_a_direct_call_raises(facade, code, status, error):
    detail = f"MCAS scorer: the system failed in the {code} way"
    exc = failing(facade, 502, {"error": code, "status": status, "detail": detail})
    if error is EndpointError:
        assert type(exc) is EndpointError
    else:
        assert isinstance(exc, error)
    assert str(exc) == detail
    assert exc.status == status


def test_the_facade_errors_map_the_platforms_codes():
    assert c.FACADE_ERRORS == {"auth": EndpointAuthError, "not_found": EndpointNotFound, "timeout": EndpointTimeout,
                               "bad_response": EndpointBadResponse, "blocked": BlockedAddress}


def test_a_facade_error_without_detail_says_where_it_failed(facade):
    exc = failing(facade, 502, {"error": "timeout"})
    assert isinstance(exc, EndpointTimeout)
    assert str(exc) == "MCAS scorer: the system under test failed behind the platform (timeout)"
    assert exc.status is None
    assert failing(facade, 502, {"error": "timeout", "status": True}).status is None


def test_one_attempt_the_platform_retries(facade):
    waited = []
    facade.route(ASK_PATH, (502, {"error": "timeout"}), (200, OK))
    cl = client(sleep=waited.append)
    with pytest.raises(EndpointTimeout):
        cl.ask(ROW)
    assert len(asks(facade)) == 1
    assert waited == []


def test_a_revoked_pass_is_an_auth_error(facade):
    exc = failing(facade, 401, {"detail": "a run key is needed"})
    assert isinstance(exc, EndpointAuthError)
    assert str(exc) == "MCAS scorer: the platform refused the run key (401); the run's pass may have ended"
    assert exc.status is None


PLATFORM_STATUSES = [
    (404, {"detail": "no route"}, EndpointNotFound, "MCAS scorer: the platform has no endpoint for this connection (404)"),
    (410, {"detail": "gone"}, EndpointNotFound, "MCAS scorer: the connection was deleted (410)"),
    (400, {"detail": "history[0]: a message is an object with a role"}, c.__dict__.get("BadRequest", EndpointError),
     "MCAS scorer: the platform refused the request (400): history[0]: a message is an object with a role"),
    (503, {}, EndpointError, "MCAS scorer: the platform answered 503"),
]


@pytest.mark.parametrize("status, body, error, message", PLATFORM_STATUSES)
def test_platform_404_410_400_and_others(facade, status, body, error, message):
    exc = failing(facade, status, body)
    if status == 400:
        assert isinstance(exc, c.BadRequest)
    assert isinstance(exc, error)
    assert str(exc) == message


@pytest.mark.parametrize("body", [[], {"refused": False}, b"not json"])
def test_a_200_that_is_not_an_answer_is_a_bad_response(facade, body):
    exc = failing(facade, 200, body)
    assert isinstance(exc, EndpointBadResponse)
    assert str(exc) == "MCAS scorer: the platform's answer is not an answer"


def test_a_redirect_from_the_platform_is_not_followed(facade):
    facade.route("/x", (200, OK))
    exc = failing(facade, 302, "http://elsewhere/x")
    assert isinstance(exc, EndpointError)
    assert "redirect (302)" in str(exc)
    assert len(asks(facade)) == 1


def test_the_platform_unreachable_at_ask(facade):
    cl = client()
    facade.stop()
    facade.server.server_close()                      # nothing listens any more
    with pytest.raises(EndpointError) as raised:
        cl.ask(ROW)
    assert str(raised.value) == "MCAS scorer: the platform could not be reached to ask 'mcas-score'"


def test_the_facade_timeout_covers_the_platforms_retries():
    def timeout(seconds):
        return c.facade_timeout_s(Descriptor(name="n", label="l", kind="rest", base_url="http://x", timeout_s=seconds))

    assert timeout(5) == 35
    assert timeout(60) == 255


# what the run records

def test_the_assessed_connection_is_recorded_as_before(facade):
    plugin = target_plugin()
    cl = EndpointClient.for_target(plugin)
    cl.ask(ROW)
    cl.ask(ROW)
    assert list(plugin.artifacts) == ["connection-mcas-score.json"]
    assert plugin.uploads == ["connection-mcas-score.json"]
    raw = plugin.artifacts["connection-mcas-score.json"]
    doc = json.loads(raw)
    that = dict(WITHOUT_SECRET, base_url=facade.base, protocol_version=None)
    assert doc["connection"] == that
    assert doc["sha256"] == hashlib.sha256(json.dumps(that, sort_keys=True).encode()).hexdigest()
    assert doc["sha256"] == Descriptor.from_dict(dict(that, secret="s3cr3t")).fingerprint()
    assert RUN_KEY.encode() not in raw


def test_nothing_is_recorded_when_the_first_ask_fails(facade):
    facade.route(ASK_PATH, (502, {"error": "auth"}))
    plugin = target_plugin()
    cl = EndpointClient.for_target(plugin)
    with pytest.raises(EndpointAuthError):
        cl.ask(ROW)
    assert plugin.artifacts == {}


ERROR_ANSWERS = ([(502, {"error": code, "status": status, "detail": f"MCAS scorer: {code}"})
                  for code, status, _ in FACADE_CODES]
                 + [(401, {"detail": "a run key is needed"})]
                 + [(status, body) for status, body, _, _ in PLATFORM_STATUSES])


@pytest.mark.parametrize("status, body", ERROR_ANSWERS)
def test_the_run_key_is_in_no_error_message(facade, status, body):
    assert RUN_KEY not in str(failing(facade, status, body))


# the run key answer

@pytest.mark.parametrize("where", ["other host", "other port", "https"])
def test_an_ask_url_off_the_platform_is_refused(facade, where):
    port = facade.port
    ask_url = {"other host": f"http://evil.example:{port}{ASK_PATH}",
               "other port": f"http://127.0.0.1:{port + 1}{ASK_PATH}",
               "https": f"https://127.0.0.1:{port}{ASK_PATH}"}[where]
    facade.route(RUN_KEYS, (201, issued(facade, ask_url=ask_url)))
    with pytest.raises(EndpointError, match="is not on PLATFORM_URL"):
        client()
    assert [r["path"] for r in facade.seen] == [RESOLVE, RUN_KEYS]


def test_a_run_key_answer_without_an_ask_url_is_refused(facade):
    facade.route(RUN_KEYS, (201, issued(facade, endpoints={"openai": {"base_url": facade.base + "/v1",
                                                                       "model": "mcas-score"}})))
    with pytest.raises(EndpointError) as raised:
        client()
    assert str(raised.value) == "the platform's run key answer for 'mcas-score' has no aisc ask_url"
    facade.route(RUN_KEYS, (201, issued(facade, key="")))
    with pytest.raises(EndpointError) as raised:
        client()
    assert str(raised.value) == "the platform's run key answer for 'mcas-score' has no key"


def test_a_refused_run_key_is_an_endpoint_error(facade):
    facade.route(RUN_KEYS, (403, {"detail": "not in this run's pass"}))
    with pytest.raises(EndpointError) as raised:
        client()
    assert str(raised.value) == "the platform refused a run key for 'mcas-score': not in this run's pass"
    facade.route(RUN_KEYS, (404, {"detail": "Scoring engine has no endpoint", "reason": "no_endpoint"}))
    with pytest.raises(TargetHasNoEndpoint):
        client()


def test_a_target_without_an_endpoint_asks_for_no_run_key(facade):
    facade.route(RESOLVE, (404, {"detail": "Scoring engine has no endpoint", "reason": "no_endpoint"}))
    with pytest.raises(TargetHasNoEndpoint):
        client()
    assert [r for r in facade.seen if r["path"] == RUN_KEYS] == []


def test_for_input_under_a_pass_uses_the_connection_routes(facade):
    resolve = "/internal/projects/pid-1/connections/mcas-chat"
    facade.route(resolve, (200, resolved(facade, name="mcas-chat")))
    facade.route(resolve + "/run-keys", (201, issued(facade, connection="mcas-chat")))
    EndpointClient.for_input(FakePlugin({"system": {"value": "connection:pid-1/mcas-chat"}}), "system")
    assert [r["path"] for r in facade.seen] == [resolve, resolve + "/run-keys"]


# without a pass

def test_without_a_pass_the_client_calls_directly(facade, monkeypatch):
    monkeypatch.delenv("PLATFORM_RUN_PASS")
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    facade.route(RESOLVE, (200, dict(SCORER, base_url=facade.base, allowed_hosts=[facade.host], denied_addresses=[])))
    cl = client()
    assert cl.run_key is None
    assert cl.descriptor.secret == "s3cr3t"
    assert [r for r in facade.seen if r["path"] == RUN_KEYS] == []
    assert cl.ask(ROW).text == OUTPUT
    assert [r["path"] for r in facade.seen][-1] == "/score"
    assert asks(facade) == []
    assert EndpointClient(Descriptor.from_dict(dict(SCORER, base_url=facade.base))).run_key is None


# tools

TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {}}}]


def test_tools_are_refused_before_the_call_when_the_endpoint_does_not_carry_them(facade):
    with pytest.raises(EndpointError) as raised:
        client().ask("q", tools=TOOLS)
    assert str(raised.value) == "MCAS scorer: this endpoint does not carry tools (it carries text, history)"
    assert asks(facade) == []
    facade.route(RUN_KEYS, (201, issued(facade, capabilities=("text", "history", "tools"))))
    client().ask("q", tools=TOOLS)
    (req,) = asks(facade)
    assert json.loads(req["body"])["params"] == {"tools": TOOLS}


# hardening: the key leaves no trace, the ask URL carries no userinfo, the run key answer is JSON

def frame_locals(exc):
    """The repr of every local of every library frame the exception (and the ones it chains) keeps."""
    seen, out = set(), []
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        tb = exc.__traceback__
        while tb is not None:
            frame = tb.tb_frame
            if "aisc_plugin_interface" in frame.f_code.co_filename:
                for name, value in frame.f_locals.items():
                    try:
                        out.append(f"{name}={value!r}")
                    except Exception:
                        pass
            tb = tb.tb_next
        exc = exc.__cause__ or exc.__context__
    return "\n".join(out)


@pytest.mark.parametrize("failure", ["run key refused", "ask url off the platform", "no endpoint",
                                     "run key answer without a key"])
def test_a_secret_the_platform_still_returns_is_in_no_frame_of_a_run_key_failure(facade, failure):
    facade.route(RESOLVE, (200, resolved(facade, secret="s3cr3t")))
    facade.route(RUN_KEYS, {
        "run key refused": (403, {"detail": "not in this run's pass"}),
        "ask url off the platform": (201, issued(facade, ask_url=f"http://evil.example:{facade.port}{ASK_PATH}")),
        "no endpoint": (404, {"detail": "Scoring engine has no endpoint", "reason": "no_endpoint"}),
        "run key answer without a key": (201, issued(facade, key="")),
    }[failure])
    with pytest.raises(EndpointError) as raised:
        client()
    assert "s3cr3t" not in frame_locals(raised.value)


@pytest.mark.parametrize("ask_url", [
    "http://evil.example@{host}" + ASK_PATH,
    "http://evil.example\\@{host}" + ASK_PATH,
    "http://user:pw@{host}" + ASK_PATH,
    "http://user@{host}" + ASK_PATH,
])
def test_an_ask_url_with_userinfo_is_refused(facade, ask_url):
    facade.route(RUN_KEYS, (201, issued(facade, ask_url=ask_url.format(host=facade.host))))
    with pytest.raises(EndpointError, match="is not on PLATFORM_URL"):
        client()
    assert [r["path"] for r in facade.seen] == [RESOLVE, RUN_KEYS]


@pytest.mark.parametrize("ask_url", [
    "http://evil.example＠{host}" + ASK_PATH,
    "http://evil.example﹫{host}" + ASK_PATH,
    "http://[127.0.0.1:{port}" + ASK_PATH,
], ids=["fullwidth at sign", "small at sign", "unclosed ipv6 bracket"])
def test_an_ask_url_the_parser_cannot_read_is_refused_like_another_host(facade, ask_url):
    facade.route(RUN_KEYS, (201, issued(facade, ask_url=ask_url.format(host=facade.host, port=facade.port))))
    with pytest.raises(EndpointError, match="is not on PLATFORM_URL"):
        client()
    assert [r["path"] for r in facade.seen] == [RESOLVE, RUN_KEYS]


PLATFORM_NAME_WITH_K = "kelvin.test"


@pytest.fixture
def platform_name_with_k(facade, monkeypatch):
    """PLATFORM_URL names the platform by a host with a k, which resolves only to the stub."""
    real = c.socket.getaddrinfo

    def getaddrinfo(host, *args, **kwargs):
        return real("127.0.0.1" if host == PLATFORM_NAME_WITH_K else host, *args, **kwargs)

    monkeypatch.setattr(c.socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setenv("PLATFORM_URL", f"http://{PLATFORM_NAME_WITH_K}:{facade.port}")
    return facade


@pytest.mark.parametrize("ask_url", [
    "http://{host}" + ASK_PATH + "\uff20caf\u00e9",
    "http://\u212aelvin.test:{port}" + ASK_PATH,
    "http://{host}\r" + ASK_PATH,
    "http://{host}" + ASK_PATH.replace("mcas-score", "mcas score"),
    "http://{host}" + ASK_PATH.replace("mcas-score", "mcas\tscore"),
], ids=["non-ascii in the path", "kelvin sign host", "carriage return after the port", "space in the path",
        "tab in the path"])
def test_an_ask_url_that_is_not_plain_ascii_is_refused_like_another_host(platform_name_with_k, ask_url):
    facade = platform_name_with_k
    url = ask_url.format(host=f"{PLATFORM_NAME_WITH_K}:{facade.port}", port=facade.port)
    facade.route(RUN_KEYS, (201, issued(facade, ask_url=url)))
    with pytest.raises(EndpointError, match="is not on PLATFORM_URL"):
        client()
    assert [r["path"] for r in facade.seen] == [RESOLVE, RUN_KEYS]


def test_the_changelog_security_entry_says_ask_urls_not_plain_ascii_are_refused():
    changelog = (Path(__file__).parents[1] / "CHANGELOG.md").read_text(encoding="utf-8")
    release = changelog.split("\n## ")[1]
    assert release.startswith("2026-10-25: 0.4.0")
    security = release.split("\nSecurity\n", 1)[1].split("\n\n", 1)[0]
    assert re.search(r"not plain (printable )?ASCII", security), security
    assert "cannot read" in security, security


@pytest.mark.parametrize("body", [b"not json", b"<html>gateway</html>", b"", b"[]", b'"mcas-score"'])
def test_a_resolve_answer_that_is_not_a_json_object_is_an_endpoint_error(facade, body):
    facade.route(RESOLVE, (200, body))
    with pytest.raises(EndpointBadResponse, match=f"resolving connection {KEY!r} was not a JSON object"):
        client()
    assert [r["path"] for r in facade.seen] == [RESOLVE]


@pytest.mark.parametrize("body", [b"[]", b'"aisc-run-abc"', b"null"])
def test_a_run_key_answer_that_is_not_a_json_object_is_a_bad_response(facade, body):
    facade.route(RUN_KEYS, (201, body))
    with pytest.raises(EndpointBadResponse, match="run key answer for 'mcas-score' is not a JSON object"):
        client()
    assert asks(facade) == []


@pytest.mark.parametrize("body", [b"not json", b"<html>gateway</html>", b""])
def test_a_run_key_answer_that_is_not_json_is_an_endpoint_error(facade, body):
    facade.route(RUN_KEYS, (201, body))
    with pytest.raises(EndpointError, match="the platform's run-key answer was not JSON"):
        client()
    assert asks(facade) == []
