"""
Entry point to run the fake server. From fakeserver/ directory:
  uv run python run.py
  uv run python run.py --port 9000 --sleep 0.3
"""
from fakeserver.server import main

if __name__ == "__main__":
    main()
