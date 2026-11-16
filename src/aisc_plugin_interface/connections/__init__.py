"""Calling a system under test that a project registered under Manage → Connections.

A connection lives on the platform (schema ``connection`` of the project's database). An evaluation
binds it as a ``resource`` input whose value is ``connection:<project pid>/<name>``. A plugin asks
``EndpointClient.for_input(self, "<input name>")`` (or ``for_target``) for a client, then
``client.ask(prompt)``. It records what it assessed as the run artifact ``connection-<name>.json``
(never the key).

Two ways the client reaches the system:
- **Through the platform**, when the run has a pass (``PLATFORM_RUN_PASS``): the client resolves the
  connection without its key (a key the platform still returns is dropped), takes a run key for it and
  asks the platform's ``aisc/ask``, which holds the key, renders the request, calls the system with its
  retries and reads the answer. The client makes one attempt; the platform's error codes are raised as
  the exception classes a direct call raises (``FACADE_ERRORS``). The ask URL must be on
  ``PLATFORM_URL``'s origin, or the run key is not sent.
- **Directly**, for a client made by hand from a ``Descriptor`` and one resolved with the service token
  of older platforms and setups (``PLATFORM_CONNECTIONS_TOKEN``, no pass): the client renders the
  request, calls the system, reads the answer and turns a declared refusal into an answer marked
  ``refused``, with the rules below.

Outbound calls are refused to loopback, private, link-local, multicast, reserved and unspecified
addresses, to any hostname that resolves to one, to schemes other than http(s), and on redirects.
For a client resolved through the platform (``for_input``, ``for_target``) the internal hosts that
are allowed come from the platform with the connection: the platform's own ``CONNECTIONS_ALLOWED_HOSTS``
plus the project's Manage, Connections, Allowed internal hosts, minus the addresses it never allows
(the stack's services, cloud metadata). ``CONNECTIONS_ALLOWED_HOSTS`` in the run's own environment
opens nothing for such a client; it is read only by a direct ``call()`` given no ``allowed_hosts``.
This protects honest plugins from a misconfigured connection; it is not a sandbox for plugin code.

Standard library only, so no plugin gains a dependency.

The package has one module per job: ``render`` (the request and the answer, as data), ``transport`` (the guarded
call), ``platform_client`` (the run key and the EndpointClient) and ``errors``. Every name the single module had is
still a name of the package; a test that replaces one replaces it in the module that uses it.
"""
from __future__ import annotations

# ruff: noqa: F401  (every name below is re-exported: the package keeps the single module's names)

# the tests reach the resolver through the package (connections.socket.getaddrinfo), as they did the module
import socket

from .errors import (BadReference, BadRequest, BlockedAddress, EndpointAuthError, EndpointBadResponse,
                     EndpointError, EndpointNotFound, EndpointTimeout, FACADE_ERRORS, is_no_endpoint, NoEndpoint,
                     TargetHasNoEndpoint)
from .render import (_A2A_DONE, _a2a_legacy, _a2a_outcome, _a2a_part, a2a_request, Answer, check_history,
                     _check_input, _check_message, check_params, Descriptor, _ends_with_tool_results, extract,
                     _good_call, HISTORY_ROLES, MAX_HISTORY_BYTES, MAX_HISTORY_MESSAGES, MAX_PARAMS_BYTES,
                     _MESSAGE_FIELDS, _NAME, _object, _openai_answer, parse_reference, _parts_value, _PATH_PART,
                     _PLACEHOLDER, _platform_answer, reference_of, REFERENCE_PREFIX, _refusal_of, render_request,
                     _render_value, _scrub, _scrub_all, _secrets_of, _tool_calls_of)
from .transport import (_a2a_answer, _addresses, allowed_hosts_from_env, _as_ipv4, call, _check_address,
                        _check_peer, checked_addresses, guard_url, _GuardedHTTPConnection, _GuardedHTTPHandler,
                        _GuardedHTTPSConnection, _GuardedHTTPSHandler, _is_loopback, _listed, MAX_ANSWER_BYTES,
                        _NoRedirect, _OPENER, _POLICY, _Policy, _post_json, _read_capped, _Redirected, _refused_ip,
                        RETRY_WAITS, _to_proxy)
from .platform_client import (_checked_run_key, EndpointClient, facade_timeout_s, _issue_run_key, _on_platform,
                              _origin, platform_auth, PLATFORM_AUTH_NEEDED, _platform_error, _run_key_from, RunKey,
                              _URL_ROLES)
