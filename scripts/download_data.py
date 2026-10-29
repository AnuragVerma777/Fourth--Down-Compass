"""
scripts/download_data.py
────────────────────────
Download nflverse play-by-play parquet files (2020-2025).
Idempotent: skips files that already exist.
"""
from pathlib import Path
import urllib.request, sys

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://github.com/nflverse/nflverse-data/releases/download/pbp"
YEARS = range(2020, 2026)


def download():
    print("=" * 60)
    print("NFLVERSE PLAY-BY-PLAY DOWNLOAD  (2020 -> 2025)")
    print(f"Destination: {DATA_DIR}")
    print("=" * 60)
    for yr in YEARS:
        name = f"play_by_play_{yr}.parquet"
        dest = DATA_DIR / name
        if dest.exists():
            mb = dest.stat().st_size / 1_048_576
            print(f"  [EXISTS]  {name}  ({mb:.1f} MB)")
            continue
        url = f"{BASE}/{name}"
        print(f"  [GET]     {yr} ... ", end="", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            mb = dest.stat().st_size / 1_048_576
            print(f"OK  ({mb:.1f} MB)")
        except Exception as exc:
            print(f"FAILED -> {exc}")
            sys.exit(1)
    print("Done.\n")


if __name__ == "__main__":
    download()
