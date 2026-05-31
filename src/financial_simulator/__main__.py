"""Support running the CLI via `python -m financial_simulator`.

This delegates to the Typer application defined in cli.py.
"""

from .cli import main

if __name__ == "__main__":
    main()
