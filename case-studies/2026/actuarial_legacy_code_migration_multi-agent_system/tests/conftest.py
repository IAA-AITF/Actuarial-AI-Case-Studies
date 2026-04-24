"""
Custom pytest plugin that writes a structured JSON report for every test.

Each entry records:
  - test_id        : fully qualified node-id
  - description    : first line of the test's docstring (or "No description")
  - category       : "data" or "content" (parsed from a @pytest.mark)
  - expected       : value declared in the docstring's "Expected:" line
  - actual         : value captured from the docstring's "Actual:" line or from
                     assertion introspection
  - status         : PASSED / FAILED / ERROR
  - detail         : short failure message (empty on pass)

The report is written to <test-dir>/test_results.json after the session ends.
"""

import json
import os
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "data: marks a test as a data/format test")
    config.addinivalue_line("markers", "content: marks a test as a content/numerical test")


class _ResultCollector:
    def __init__(self):
        self.results: list[dict] = []


collector = _ResultCollector()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    # Decide whether this phase report is recordable. The call phase is
    # always recorded (PASSED / FAILED / ERROR). The setup and teardown
    # phases are only recorded on failure, with status ERROR, so the JSON
    # report reflects fixture errors (which would otherwise be invisible
    # because they never reach the call phase).
    if report.when == "call":
        if report.passed:
            status = "PASSED"
            detail = ""
        elif report.failed:
            status = "FAILED"
            detail = str(report.longrepr)[:2000]
        else:
            status = "ERROR"
            detail = str(report.longrepr)[:2000]
    elif report.when in ("setup", "teardown") and not report.passed:
        status = "ERROR"
        detail = (str(report.longrepr)[:2000]
                  if report.longrepr else f"{report.when} phase error")
    else:
        return

    # ── description from docstring ──
    docstring = (item.function.__doc__ or "").strip()
    description = docstring.split("\n")[0] if docstring else "No description"

    # ── expected / actual from docstring markers ──
    expected = ""
    actual = ""
    for line in docstring.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("expected:"):
            expected = stripped.split(":", 1)[1].strip()
        if stripped.lower().startswith("actual:"):
            actual = stripped.split(":", 1)[1].strip()

    # ── category from pytest.mark ──
    if any(m.name == "data" for m in item.iter_markers()):
        category = "data"
    elif any(m.name == "content" for m in item.iter_markers()):
        category = "content"
    else:
        category = "uncategorized"

    # ── de-duplicate by test_id so a setup-phase ERROR is not later
    #    doubled by a call-phase record for the same node ──
    if any(r["test_id"] == report.nodeid for r in collector.results):
        return

    collector.results.append({
        "test_id": report.nodeid,
        "description": description,
        "category": category,
        "expected": expected,
        "actual": actual,
        "status": status,
        "detail": detail,
    })


def pytest_sessionfinish(session, exitstatus):
    """Write the JSON report next to the test files."""
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_results.json",
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(collector.results, f, indent=2)
