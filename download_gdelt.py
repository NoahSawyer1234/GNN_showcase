"""Download raw GDELT event files for a date range.

  python download_gdelt.py --start 2025-01-01 --end 2025-03-31 --out data/raw/gdelt

GDELT 1.0 (default): one file per day, http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip
  (daily files exist from 2013-04-01 onwards).
GDELT 2.0 (--version 2): one file per 15 minutes, http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.export.CSV.zip
  (from 2015-02-18). A day is 96 files, so use --step-minutes 60 (or more) to thin them out.
  NOTE: the 1.0 URL pattern was checked against GDELT's site; the 2.0 pattern is from GDELT's
  documentation and has not been verified from this environment.

Existing files are skipped, so the script can be re-run to resume. Missing files (HTTP 404) are logged.
"""
import argparse
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

BASE = {1: "http://data.gdeltproject.org/events", 2: "http://data.gdeltproject.org/gdeltv2"}


def urls(version, start, end, step_minutes):
    if version == 1:
        d = start
        while d <= end:
            yield f"{BASE[1]}/{d:%Y%m%d}.export.CSV.zip"
            d += timedelta(days=1)
    else:
        t = start
        stop = end + timedelta(days=1)
        while t < stop:
            yield f"{BASE[2]}/{t:%Y%m%d%H%M%S}.export.CSV.zip"
            t += timedelta(minutes=step_minutes)


def fetch(url, dest: Path, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dest.with_suffix(".part"), "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            dest.with_suffix(".part").rename(dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            err = e
        except Exception as e:                       # network hiccup
            err = e
        time.sleep(2 ** attempt)
    print(f"  failed after {retries} tries: {url} ({err})")
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--out", default="data/raw/gdelt")
    p.add_argument("--version", type=int, choices=[1, 2], default=1)
    p.add_argument("--step-minutes", type=int, default=60,
                   help="v2 only: spacing between files, a multiple of 15")
    args = p.parse_args()
    if args.version == 2 and args.step_minutes % 15:
        p.error("--step-minutes must be a multiple of 15")

    start, end = (datetime.strptime(s, "%Y-%m-%d") for s in (args.start, args.end))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = list(urls(args.version, start, end, args.step_minutes))
    got = skipped = missing = 0
    for i, url in enumerate(todo, 1):
        dest = out / url.rsplit("/", 1)[1]
        if dest.exists():
            skipped += 1
        elif fetch(url, dest):
            got += 1
        else:
            missing += 1
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  downloaded={got} existing={skipped} missing={missing}")


if __name__ == "__main__":
    main()
