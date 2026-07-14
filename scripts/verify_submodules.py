"""Fail fast when the local upstream backends do not match EvoWorld pins."""

from __future__ import annotations

from pathlib import Path
import sys

from evoworld.adapters.revisions import SubmoduleAuditError, inspect_submodules


def main() -> int:
    try:
        revisions = inspect_submodules(Path(__file__).resolve().parents[1])
    except SubmoduleAuditError as error:
        print(f"submodule verification failed: {error}", file=sys.stderr)
        return 1
    for path, revision in revisions.items():
        print(f"{path}: {revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
