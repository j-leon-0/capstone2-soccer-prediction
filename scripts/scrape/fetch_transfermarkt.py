"""Fetch season-start squad market values from Transfermarkt.

For each season the competition "startseite" (overview) page lists every Premier
League club with its total squad market value at the start of that season -- a
clean proxy for squad quality / cumulative transfer spend that is fixed before a
ball is kicked, so it is strictly pre-match.

URL: /premier-league/startseite/wettbewerb/GB1/plus/?saison_id=<startYear>

Transfermarkt is bot-protected and intermittently returns 502/000, so each fetch
is retried. Raw HTML is cached under data/raw/external/transfermarkt/.

Output
------
* ``data/processed/external/epl_squad_value.csv`` with columns
  ``Season, Team, squad_value_m, value_rank, value_share`` (share of season total).
"""

from __future__ import annotations

import html as html_lib
import re
import time
from pathlib import Path

import pandas as pd
import urllib.request
import urllib.error

CACHE_DIR = Path("data/raw/external/transfermarkt")
OUTPUT_PATH = Path("data/processed/external/epl_squad_value.csv")

SEASONS = list(range(2016, 2026))  # start year; 2016 -> 2016_2017 ... 2025 -> 2025_2026
URL = ("https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1"
       "/plus/?saison_id={year}")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.transfermarkt.com/",
}

# Transfermarkt club names -> our dataset's short names.
TM_TO_OURS = {
    "Chelsea FC": "Chelsea", "Manchester City": "Man City",
    "Tottenham Hotspur": "Tottenham", "Arsenal FC": "Arsenal",
    "Manchester United": "Man United", "Liverpool FC": "Liverpool",
    "Everton FC": "Everton", "West Ham United": "West Ham",
    "Southampton FC": "Southampton", "Crystal Palace": "Crystal Palace",
    "Leicester City": "Leicester", "Stoke City": "Stoke",
    "Watford FC": "Watford", "Swansea City": "Swansea", "Hull City": "Hull",
    "Sunderland AFC": "Sunderland", "AFC Bournemouth": "Bournemouth",
    "Middlesbrough FC": "Middlesbrough", "West Bromwich Albion": "West Brom",
    "Burnley FC": "Burnley", "Norwich City": "Norwich",
    "Brighton & Hove Albion": "Brighton", "Huddersfield Town": "Huddersfield",
    "Cardiff City": "Cardiff", "Fulham FC": "Fulham",
    "Wolverhampton Wanderers": "Wolves", "Sheffield United": "Sheffield United",
    "Aston Villa": "Aston Villa", "Leeds United": "Leeds",
    "Brentford FC": "Brentford", "Nottingham Forest": "Nott'm Forest",
    "Luton Town": "Luton", "Ipswich Town": "Ipswich",
    "Newcastle United": "Newcastle",
}


def eur_to_millions(s: str) -> float:
    num = float(re.sub(r"[€,]", "", s[:-2] if s.endswith("bn") else s[:-1]))
    if s.endswith("bn"):
        return num * 1000.0
    if s.endswith("m"):
        return num
    if s.endswith("k"):
        return num / 1000.0
    return num


def fetch_html(year: int, refresh: bool = False, retries: int = 6) -> str:
    cache = CACHE_DIR / f"GB1_{year}.html"
    if cache.exists() and not refresh and cache.stat().st_size > 50000:
        return cache.read_text()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    last = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL.format(year=year), headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            if len(html) > 50000:
                cache.write_text(html)
                return html
            last = f"short ({len(html)})"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last = str(exc)
        time.sleep(2 + attempt)
    raise RuntimeError(f"Failed to fetch {year}: {last}")


def parse_clubs(html: str, year: int) -> list[tuple[str, float]]:
    rows = re.split(r"<tr[ >]", html)
    out = []
    name_re = re.compile(
        r'<a title="([^"]+)" href="/[a-z0-9-]+/startseite/verein/\d+/saison_id/'
        + str(year) + '"')
    for r in rows:
        m = name_re.search(r)
        if not m:
            continue
        vals = re.findall(r"€[\d.,]+(?:bn|m|k)", r)
        if not vals:
            continue
        out.append((html_lib.unescape(m.group(1)), eur_to_millions(vals[-1])))
    return out


def main() -> None:
    records = []
    for year in SEASONS:
        season = f"{year}_{year + 1}"
        html = fetch_html(year)
        clubs = parse_clubs(html, year)
        if len(clubs) != 20:
            print(f"  WARNING {season}: parsed {len(clubs)} clubs (expected 20)")
        total = sum(v for _, v in clubs)
        ranked = sorted(clubs, key=lambda x: -x[1])
        for rank, (name, val) in enumerate(ranked, start=1):
            team = TM_TO_OURS.get(name, name)
            records.append({
                "Season": season, "Team": team, "squad_value_m": round(val, 2),
                "value_rank": rank, "value_share": round(val / total, 4),
            })
        print(f"  {season}: {len(clubs)} clubs, total €{total/1000:.2f}bn")

    df = pd.DataFrame(records)
    unmapped = sorted(set(df["Team"]) - set(TM_TO_OURS.values()))
    if unmapped:
        print("Unmapped club names (check TM_TO_OURS):", unmapped)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
