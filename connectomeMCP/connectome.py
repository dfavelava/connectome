import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from connectomemcp import main, mcp

__all__ = ["main", "mcp"]


if __name__ == "__main__":
    main()
