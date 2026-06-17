"""Conformance tests for the OkxClient facade (indicator/okx/client.py).

The facade is the single object the LIVE executor holds as self._client
(wired in runner.py: OkxClient(cfg) -> V7OkxExecutor(client=...)).  The
trading unit tests inject a MagicMock as the client, and a MagicMock
auto-vivifies ANY attribute -- so a method the executor calls but the
facade forgot to expose stays GREEN in the suite yet raises AttributeError
in live (swallowed by the executor's bare except -> silent "OPEN ABORTED").

This exact bug class has shipped TWICE:
  - amend_algo_stop dropped the inst_id kwarg (2026-06-10) -> trailing stop
    never moved on the exchange for ~2 weeks.
  - set_leverage was never added to the facade (2026-06-17) -> every
    isolated-margin open aborted; the system could not open any position.

The AST test below fails the instant the executor calls a self._client
method the facade does not define, and the signature test fails if a facade
method drops a parameter its REST counterpart accepts -- catching the whole
class cheaply, without hand-maintaining a method list.
"""
from __future__ import annotations

import ast
import inspect

from indicator.okx import executor as executor_mod
from indicator.okx.client import OkxClient
from indicator.okx.rest import OkxRestClient


def _methods_called_on_self_client() -> set[str]:
    """Every attribute accessed as ``self._client.<name>`` in executor.py."""
    tree = ast.parse(inspect.getsource(executor_mod))
    names: set[str] = set()
    for node in ast.walk(tree):
        # match the outer Attribute of  self._client.<name>
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "_client"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "self"
        ):
            names.add(node.attr)
    return names


def test_ast_matcher_finds_client_calls():
    # Guardrail: if the matcher silently breaks (e.g. executor renames the
    # attribute), the conformance test below would pass vacuously.  Assert it
    # actually found the call surface first.
    called = _methods_called_on_self_client()
    assert "set_leverage" in called and "submit_market_order" in called, (
        "AST matcher found no/incomplete self._client.* usage in executor.py "
        f"-- matcher is broken, not the facade. Found: {sorted(called)}"
    )


def test_facade_exposes_every_method_executor_calls():
    called = _methods_called_on_self_client()
    missing = sorted(
        name for name in called
        if not callable(getattr(OkxClient, name, None))
    )
    assert not missing, (
        "OkxClient facade is MISSING methods the live executor calls on "
        f"self._client: {missing}. Add a passthrough in indicator/okx/client.py "
        "mirroring the OkxRestClient signature (see the amend_algo_stop / "
        "set_leverage facade-skip bugs)."
    )


def test_facade_signatures_match_rest_for_shared_methods():
    """A facade method must accept every parameter its REST counterpart does.

    The amend_algo_stop bug (2026-06-10) was a facade method that EXISTED but
    silently dropped the inst_id kwarg, so the executor's call raised TypeError
    (swallowed) and the trailing stop never moved.  A missing-method test alone
    would not have caught it -- this signature-superset check does.
    """
    called = _methods_called_on_self_client()
    drift: dict[str, list[str]] = {}
    for name in sorted(called):
        facade_fn = getattr(OkxClient, name, None)
        rest_fn = getattr(OkxRestClient, name, None)
        if facade_fn is None or rest_fn is None:
            # Not a REST-backed method (start_ws / close / connectivity /
            # subscribe_* live on the WS side or the facade itself).
            continue
        facade_params = set(inspect.signature(facade_fn).parameters)
        rest_params = set(inspect.signature(rest_fn).parameters)
        dropped = sorted(rest_params - facade_params)
        if dropped:
            drift[name] = dropped
    assert not drift, (
        "OkxClient facade signatures DRIFT from OkxRestClient (facade drops "
        f"params the REST layer accepts): {drift}. Keep the facade in lockstep."
    )
