"""Shared pytest configuration: Hypothesis profiles for the fuzz tests.

Select one with HYPOTHESIS_PROFILE (default: "default").
"""

import os

try:
    from hypothesis import settings
except ImportError:  # the fuzz tests skip themselves without hypothesis
    pass
else:
    # tsfresh is slow, so no per-example deadline
    settings.register_profile("default", max_examples=200, deadline=None)
    settings.register_profile("ci", max_examples=10, deadline=None)
    # for hunting rare cases in a few features: pytest -m fuzz -k <feature>
    settings.register_profile("thorough", max_examples=5000, deadline=None)
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
