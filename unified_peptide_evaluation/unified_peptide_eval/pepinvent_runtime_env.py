#!/usr/bin/env python3
import os
from pathlib import Path


def make_runtime_env(base_env=None):
    env = dict(os.environ if base_env is None else base_env)
    here = str(Path(__file__).resolve().parent)

    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        here if not old else here + os.pathsep + old
    )
    env["PEPINVENT_RUNTIME_OVERRIDES"] = "1"
    return env
