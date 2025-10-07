#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Διαβάζει out/players_<SEASON>_perGame.csv και γράφει out/player_urls_<SEASON>.csv
Στήλες εισόδου (τουλάχιστον μία από κάθε ομάδα):
- player_code | playerId | id
- Player | player | name | FullName

Usage:
  python scripts/build_player_urls.py --season E2025 --competition E
Notes:
  competition: E=EuroLeague, U=EuroCup  (επηρεάζει το path: /euroleague/ ή /eurocup/)
"""

import argparse
import os
import time
import re
import unicodedata
import pandas as pd
import requests

def slugify(name: str) -> str:
    """Μετατρέπει 'Vasilije Micic' -> 'vasilije-micic' (χωρίς τόνους/σύμβολα)."""
    if not name:
        return ""
    # normalize unicode → remove accents
    name = unicodedata.normalize('NFKD', name)
    name = "".join([c for c in name if not unicodedata.combining(c)])
    name = name.lower()
    # αντικατάσταση μη αλφαριθμητικών με space
    name = re.sub(r"[^a-z0-9]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name.replace(" ", "-")

def first_last(name: str) -> str:
    if not name:
        return ""
    parts = [p for p in re.split(r"\s+", name.strip()) if p]
    if len(parts) == 1:
        return parts[0]
    # πάρε πρώτο + τελευταίο
    return f"{parts[0]} {parts[-1]}"

def choose_col(df: pd.DataFrame, options):
    for c in options:
        if c in df.columns:
            return c
    return None

def url_candidates(base_lang: str, comp_path: str, slug: str, code: str):
    # canonical
    yield f"https://www.euroleaguebasketball.net/{base_lang}/{comp_path}/players/{slug}/{code}/"
    # δοκίμασε και χωρίς trailing slash
    yield f"https://www.euroleaguebasketball.net/{base_lang}/{comp_path}/players/{slug}/{code}"
    # με πιθανό διαφορετικό slug (μόνο τελευταίο όνομα)
    last = slug.split("-")[-1] if slug else ""
    if last:
        yield f"https://www.euroleaguebasketball.net/{base_lang}/{comp_path}/players/{last}/{code}/"

def probe_url(session: requests.Session, urls):
    for u in urls:
        try:
            r = session.head(u, allow_redirects=True, timeout=12)
            # μερικές φορές το HEAD επιστρέφει 405 → κάνε GET lightweight
            if r.status_code in (403, 404, 405):
                r = session.get(u, allow_redirects=True, timeout=12)
            if 200 <= r.status_code < 400:
                return r.url  # resolved
        except requests.RequestException:
            pass
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="E2025")
    ap.add_argument("--competition", default="E", help="E=EuroLeague, U=EuroCup")
    ap.add_argument("--infile", default=None, help="override input CSV path")
    ap.add_argument("--outfile", default=None, help="override output CSV path")
    ap.add_argument("--lang", default="el", choices=["el","en"])
    ap.add_argument("--sleep", type=float, default=0.3, help="delay between requests (s)")
    args = ap.parse_args()

    season = args.season
    comp = args.competition.upper()
    comp_path = "euroleague" if comp.startswith("E") else "eurocup"
    infile = args.infile or f"out/players_{season}_perGame.csv"
    outfile = args.outfile or f"out/player_urls_{season}.csv"

    if not os.path.exists(infile):
        raise SystemExit(f"Δεν βρέθηκε input CSV: {infile}")

    df = pd.read_csv(infile)

    code_col = choose_col(df, ["player_code","playerId","PlayerId","id","ID","Code","code"])
    name_col = choose_col(df, ["Player","player","name","FullName","FULLNAME"])

    if not code_col or not name_col:
        raise SystemExit(f"Δεν βρέθηκαν απαιτούμενες στήλες στο {infile} "
                         f"(code in one of ['player_code','playerId','id']; name in ['Player','name']).")

    session = requests.Session()
    session.headers.update({"User-Agent": "eurol-url-builder/1.0", "Accept": "text/html"})

    out_rows = []
    for _, row in df[[code_col, name_col]].drop_duplicates().iterrows():
        code = str(row[code_col]).strip()
        name = str(row[name_col]).strip()
        # φτιάξε διαδοχικά slugs: full name, first+last
        slugs = []
        slug_full = slugify(name)
        if slug_full:
            slugs.append(slug_full)
        fl = slugify(first_last(name))
        if fl and fl not in slugs:
            slugs.append(fl)

        found_url = ""
        for slug in slugs:
            cands_el = list(url_candidates(args.lang, comp_path, slug, code))
            found_url = probe_url(session, cands_el)
            if found_url:
                break

        out_rows.append({"player_code": code, "Player": name, "player_url": found_url})
        time.sleep(args.sleep)

    out_df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    out_df.to_csv(outfile, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {outfile} rows={len(out_df)} with {out_df['player_url'].astype(bool).sum()} resolved URLs")

if __name__ == "__main__":
    main()
