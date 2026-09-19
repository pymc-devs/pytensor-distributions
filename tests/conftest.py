def pytest_terminal_summary(terminalreporter):
    stats = terminalreporter.stats
    skipped = stats.get("skipped", [])
    total = sum(len(v) for v in stats.values())
    n_passed = len(stats.get("passed", []))
    n_skipped = len(skipped)
    terminalreporter.write_sep("-", f"{n_passed} passed, {n_skipped} skipped ({total} total)")
