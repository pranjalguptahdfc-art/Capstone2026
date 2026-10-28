"""Pytest bootstrap: load torch before any test module runs.

On this machine (Windows, system Python 3.10, torch 2.14.0+cpu) the two
OpenMP runtimes clash: if pandas (pulled in by utils/analytics_store) loads
FIRST, the later ultralytics→torch import crashes with
"OSError [WinError 1114] ... c10.dll". The reverse order (torch, then
pandas) is fine. pytest imports conftest.py before every test module, so
importing torch here guarantees the safe order regardless of test file
ordering — without touching production code.

(Standalone runners that construct a BananaDetector before importing
analytics_store already get the safe order; only pytest, where test modules
load together, needed this.)
"""

try:
    import torch  # noqa: F401  — must precede pandas in the process
except Exception:
    # torch not installed: the pure-numpy suites still run; any detector
    # test will surface the real import error itself.
    pass
