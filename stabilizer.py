"""
Top-level convenience entry point.

Usage:
    python stabilizer.py input.mp4 output.mp4 [options]
    python stabilizer.py --help
"""
from drone_stabilizer.main import main

if __name__ == "__main__":
    main()
