"""The connections module is a package of three jobs, each small enough to read: rendering a request and reading its
answer (never opening a connection), the guarded transport, and the platform client, with the errors in a module of
their own. Every name the module had before is still there, the private ones tests patch included."""

import importlib
import importlib.util
import inspect

from aisc_plugin_interface import connections

#: the module's own names, recorded on the tree before the split (imports of standard modules left out)
NAMES = [
    "Answer", "BadReference", "BadRequest", "BlockedAddress", "Descriptor", "EndpointAuthError",
    "EndpointBadResponse", "EndpointClient", "EndpointError", "EndpointNotFound", "EndpointTimeout",
    "FACADE_ERRORS", "HISTORY_ROLES", "MAX_ANSWER_BYTES", "MAX_HISTORY_BYTES", "MAX_HISTORY_MESSAGES",
    "MAX_PARAMS_BYTES", "NoEndpoint", "PLATFORM_AUTH_NEEDED", "REFERENCE_PREFIX", "RETRY_WAITS", "RunKey",
    "TargetHasNoEndpoint", "_A2A_DONE", "_GuardedHTTPConnection", "_GuardedHTTPHandler", "_GuardedHTTPSConnection",
    "_GuardedHTTPSHandler", "_MESSAGE_FIELDS", "_NoRedirect", "_Policy", "_Redirected", "_URL_ROLES", "_a2a_answer",
    "_a2a_legacy", "_a2a_outcome", "_a2a_part", "_addresses", "_as_ipv4", "_check_address", "_check_input",
    "_check_message", "_check_peer", "_checked_run_key", "_ends_with_tool_results", "_good_call", "_issue_run_key",
    "_object", "_on_platform", "_openai_answer", "_origin", "_parts_value", "_platform_answer", "_platform_error",
    "_post_json", "_read_capped", "_refusal_of", "_refused_ip", "_render_value", "_run_key_from", "_scrub",
    "_scrub_all", "_secrets_of", "_to_proxy", "_tool_calls_of", "a2a_request", "allowed_hosts_from_env", "call",
    "check_history", "check_params", "extract", "facade_timeout_s", "guard_url", "is_no_endpoint", "parse_reference",
    "platform_auth", "reference_of", "render_request",
]
PARTS = ("render", "transport", "platform_client", "errors")


def test_every_name_of_the_module_is_still_there():
    assert [n for n in NAMES if not hasattr(connections, n)] == []


def test_the_module_is_a_package_of_three_jobs():
    assert hasattr(connections, "__path__"), "connections is still one module of over a thousand lines"
    missing = [p for p in PARTS if importlib.util.find_spec(f"aisc_plugin_interface.connections.{p}") is None]
    assert missing == []
    large = {p: n for p in PARTS
             if (n := len(inspect.getsource(importlib.import_module(f"aisc_plugin_interface.connections.{p}"))
                          .splitlines())) >= 450}
    assert large == {}


def test_render_never_opens_a_connection():
    assert hasattr(connections, "__path__"), "connections is still one module: rendering and transport are mixed"
    src = inspect.getsource(importlib.import_module("aisc_plugin_interface.connections.render"))
    assert "urllib.request" not in src and "http.client" not in src and "socket" not in src
