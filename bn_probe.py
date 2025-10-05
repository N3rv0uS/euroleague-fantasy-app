#!/usr/bin/env python3
import argparse, os, sys, time, json, re
from typing import Optional, List, Dict
import pandas as pd
import requests

def http_json(url: str, timeout: int = 25) -> dict:
    r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        # Αν είναι JSON τυλιγμένο σε JS, προσπάθησε να το απομονώσεις
        txt = r.text
        i, j = txt.find("{"), txt.rfind("}")
        if i != -1 and j != -1 and j > i:
            return json.loads(txt[i:j+1])
        raise

def unify_columns(df: pd.DataFrame) -> pd.DataFrame:
    # ταιριάζουμε τα βασικά stats
    alias = {
        "MIN":  ["MIN","Min","Time","Minutes","Min."],
        "PTS":  ["PTS","Pts","Points","P"],
        "2FG":  ["2P M/A","2PM-A","2P","2P%","2P M/A %"],
        "3FG":  ["3P M/A","3PM-A","3P","3P%","3P M/A %"],
        "FT":   ["FT M/A","FTM-A","FT","FT%","FT M/A %"],
        "O":    ["ORB","OReb","Off Reb","O"],
        "D":    ["DRB","DReb","Def Reb","D"],
        "T":    ["TRB","Tot Reb","REB","T"],
        "AST":  ["AST","Assists","A"],
        "STL":  ["STL","Steals","S"],
        "TOV":  ["TOV","TO","Turnovers"],
        "BLK_FV":["BLK","Blocks For","Blocks Made"],
        "BLK_AG":["BLK Against","Blocks Against","BA"],
        "FLS_CM":["Fouls","Fouls Committed","PF","Cm"],
        "FLS_RV":["Fouls Drawn","FD","Rv"],
        "PIR":  ["EFF","Index","PIR"],
    }
    rename = {}
    for std, candidates in alias.items():
        for c in list(df.columns):
            if str(c).strip() in candidates:
                rename[c] = std
                break
    df = df.rename(columns=rename)
    # x/y split
    def split_xy(s):
        if isinstance(s,str) and "/" in s:
            a,b = s.split("/",1)
            try: return float(a), float(b)
            except: return a,b
        return None,None
    for cat in ("2FG","3FG","FT"):
        if cat in df.columns:
            m,a = zip(*[split_xy(x) for x in df[cat].tolist()])
            df[f"{cat}_M"]=m; df[f"{cat}_A"]=a
    # standard cols
    for col in ["MIN","PTS","2FG","3FG","FT","O","D","T","AST","STL","TOV","BLK_FV","BLK_AG","FLS_CM","FLS_RV","PIR"]:
        if col not in df.columns: df[col]=pd.NA
    return df

def parse_bn_html_gamelogs(page_html: str) -> pd.DataFrame:
    # προσπαθούμε με pandas.read_html να εντοπίσουμε τον πίνακα “Statistics: Games”
    # Κρατάμε τον μεγαλύτερο πίνακα με στήλες που μοιάζουν με stats.
    tables = pd.read_html(page_html)
    best = None
    score = -1
    wanted = {"MIN","PTS","2P","3P","FT","REB","AST","STL","TO","EFF","PIR"}
    for t in tables:
        s = sum(1 for c in t.columns if any(k in str(c).upper() for k in wanted))
        if s > score:
            score = s; best = t
    if best is None: return pd.DataFrame()
    df = best.copy()
    # σβήσε συνοψιστικές γραμμές (Total/Average) αν υπάρχουν
    df = df[~df.iloc[:,0].astype(str).str.contains("Total|Average", case=False, na=False)]
    df.reset_index(drop=True, inplace=True)
    # πρόσθεσε αύξοντα αριθμό παιχνιδιού & placeholders
    df.insert(0, "game_no", range(1, len(df)+1))
    return unify_columns(df)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025")
    ap.add_argument("--competition", default="E")
    ap.add_argument("--mode", default="perGame")
    ap.add_argument("--player_code", default="", help="tag για τον παίκτη (αν έχεις)")
    ap.add_argument("--json_url", default="", help="Αν έχεις JSON endpoint του BasketNews")
    ap.add_argument("--page_url", default="", help="URL της σελίδας παίκτη (HTML fallback)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame()

    if args.json_url:
        try:
            j = http_json(args.json_url)
            # Προσαρμογή: flatten πιθανό j["data"]["games"] κ.λπ.
            # Αν δεν ξέρουμε ακριβές schema, δοκίμασε auto-normalize:
            rows = None
            if isinstance(j, dict):
                for k in ("items","data","rows","list","games","matches"):
                    if isinstance(j.get(k), list):
                        rows = j[k]; break
            if rows is None and isinstance(j, list):
                rows = j
            if not rows:
                print("[warn] JSON χωρίς αναμενόμενη λίστα. Θα κάνω normalize όλο το αντικείμενο.")
                rows = [j]
            df = pd.json_normalize(rows, sep="_")
            # Ομογενοποίηση ονομάτων αν γίνεται:
            df = unify_columns(df)
            if "game_no" not in df.columns:
                df.insert(0, "game_no", range(1, len(df)+1))
        except Exception as e:
            print(f"[warn] JSON mode failed: {e}")

    if df.empty and args.page_url:
        try:
            r = requests.get(args.page_url, headers={"User-Agent":"Mozilla/5.0"}, timeout=30)
            r.raise_for_status()
            df = parse_bn_html_gamelogs(r.text)
        except Exception as e:
            print(f"[warn] HTML mode failed: {e}")

    if df.empty:
        print("No gamelog rows parsed.")
        sys.exit(0)

    if args.player_code:
        df["player_code"] = args.player_code
    df["season"] = args.season
    df["competition"] = args.competition
    df["mode"] = args.mode

    out_csv = os.path.join(args.out, f"player_gamelogs_{args.season}_{args.mode}.csv")
    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] saved {out_csv} rows={len(df)}")

if __name__ == "__main__":
    main()
