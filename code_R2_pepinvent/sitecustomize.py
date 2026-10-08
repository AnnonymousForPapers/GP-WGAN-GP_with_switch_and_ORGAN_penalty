#!/usr/bin/env python3
"""
Loaded automatically by Python when this project directory is on PYTHONPATH.

No installed REINVENT source file is modified.

Runtime overrides are enabled only when:
    PEPINVENT_RUNTIME_OVERRIDES=1
"""

import os
import sys
import traceback

if os.environ.get("PEPINVENT_RUNTIME_OVERRIDES") == "1":
    try:
        from pepinvent_runtime_overrides import install
        install()
    except Exception:
        print(
            "[PEPINVENT-RUNTIME] Failed to install runtime overrides:",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise
