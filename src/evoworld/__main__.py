"""Allow ``python -m evoworld`` to use the public CLI contract."""

from evoworld.cli import main

raise SystemExit(main())
