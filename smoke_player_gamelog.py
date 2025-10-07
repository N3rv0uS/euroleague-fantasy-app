#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, sys, os, hashlib
from typing import List
import requests
import pandas as pd
from bs4 import BeautifulSoup

UA = "eurol-scraper/1.0 (+stats smoke)"
HDRS = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"}

def fetch_html(url: str) -> str:
    r = requests.get(url, headers=HDRS, timeout=30)
    r.raise_for_status()
    return r.text

def pick_gamelog_table(html: str) -> pd.DataFrame:
    """
    Επιλέγουμε το table που μοιάζει με gamelog:
    - έχει στήλες για ημερομηνία/αντίπαλο/λεπτά/πόντους/PIR κ.λπ.
    Χρησιμοποιούμε pandas.read_html και μετά heuristic match στα headers.
    """
    # Διαβάζουμε όλα τα tables
    dfs = pd.read_html(html)
    if not dfs:
        raise RuntimeError("Δεν βρέθηκαν <table> στο HTML")

    def norm(s): 
        return re.sub(r"\W+", "", str(s).strip().lower())

    candidates = []
    for i, df in enumerate(dfs):
        cols = [norm(c) for c in df.columns.tolist()]
        score = 0
        for key in ("pir", "λεπτα", "min", "ποντοι", "pts", "ημ", "date", "αντιπαλος", "opponent"):
            if any(key in c for c in cols):
                score += 1
        candidates.append((score, i, df))

    candidates.sort(reverse=True, key=lambda x: x[0])
    best_score, best_idx, best_df = candidates[0]
    if best_score < 3:
        raise RuntimeError("Δεν βρέθηκε πίνακας που να μοιάζει με gamelog")

    df = best_df.copy()
    # καθάρισμα headers (μερικές φορές multiindex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in t]).strip() for t in df.columns.values]
    df.columns = [str(c).strip() for c in df.columns]

    return df

def parse_minutes_to_float(s):
    s = str(s)
    if ":" in s:
        m, sec = s.split(":")[:2]
        try: return int(m) + int(sec)/60.0
        except: return None
    try: return float(s)
    except: return None

def tidy(df: pd.DataFrame) -> pd.DataFrame:
    # μετονομασίες “κλασσικών” στηλών (EL/EN)
    rename_map = {
        "Ημερομηνία": "date", "Ημ/νία": "date", "Date": "date",
        "Αντίπαλος": "opponent", "Opponent": "opponent",
        "Λεπτά": "min", "Min": "min",
        "Πόντοι": "pts", "PTS": "pts",
        "PIR": "pir",
        "Δίποντα": "2pts", "2PT": "2pts",
        "Τρίποντα": "3pts", "3PT": "3pts",
        "Βολές": "ft", "FT": "ft",
        "Επιθ. Ριμπ.": "or", "OR": "or",
        "Αμ. Ριμπ.": "dr", "DR": "dr",
        "Ριμπάουντ": "tr", "TR": "tr",
        "Αστ.": "ast", "AST": "ast",
        "Κλ.": "stl", "STL": "stl",
        "Λάθη": "to", "TO": "to",
        "Κοπ.": "blk", "BLK": "blk",
        "Κοπ. Κατ.": "blka", "BLKA": "blka",
        "Φαουλ": "fc", "Fouls": "fc",
        "Κερδ. Φάουλ": "fd", "FD": "fd",
        "Αγώνας": "match"
    }
    # Χαλαρό matching: αν υπάρχει key του map μέσα στο col name, μετονομάζουμε
    new_cols = {}
    for c in df.columns:
        target = None
        for k, v in rename_map.items():
            if k.lower() in c.lower():
                target = v; break
        new_cols[c] = target or c
    df = df.rename(columns=new_cols)

    # minutes → float
    if "min" in df.columns:
        df["min_dec"] = df["min"].apply(parse_minutes_to_float)

    return df

def main():
    if len(sys.argv) < 3:
        print("Usage: smoke_player_gamelog.py <player_url> <out_csv>")
        sys.exit(2)

    url, out_csv = sys.argv[1], sys.argv[2]
    html = fetch_html(url)
    df = pick_gamelog_table(html)
    df = tidy(df)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved gamelogs → {out_csv} (rows={len(df)})")

if __name__ == "__main__":
    main()
