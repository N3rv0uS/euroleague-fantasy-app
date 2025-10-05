#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, json, re, warnings
from typing import List, Dict, Optional
import pandas as pd
import requests

warnings.simplefilter("ignore", category=FutureWarning)

UA = {"User-Agent": "Mozilla/5.0"}

# ---------------------- Utilities ----------------------
def http_get(url: str, timeout: int = 30) -> str:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.text

def http_json(url: str, timeout: int = 30) -> dict:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        txt = r.text
        i, j = txt.find("{"), txt.rfind("}")
        if i != -1 and j != -1 and j > i:
            return json.loads(txt[i:j+1])
        raise

def clean_header_label(lbl: str) -> str:
    s = str(lbl)
    s = s.replace("\n", " ").replace("\r", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    return s

def flatten_headers(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for tup in df.columns:
            parts = [clean_header_label(p) for p in tup if p and str(p).strip() != ""]
            new_cols.append(" ".join(parts) if parts else "")
        df.columns = new_cols
    else:
        df.columns = [clean_header_label(c) for c in df.columns]
    return df

def extract_player_name_from_html(html: str) -> Optional[str]:
    # <h1>Will Clyburn #21</h1>  ή παρόμοιο
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.I|re.S)
    if not m: 
        return None
    raw = re.sub(r"<.*?>", "", m.group(1)).strip()
    raw = raw.split("#")[0].strip()
    return raw or None

# ---------------------- Column unifier ----------------------
def unify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Χαρτογράφηση BN headers (EN/GR variants) στα standard:
    MIN, PTS, 2FG, 3FG, FT, O, D, T, AST, STL, TOV, BLK_FV, BLK_AG, FLS_CM, FLS_RV, PIR.
    Περιλαμβάνει και aliases τύπου AS/ST/TO/BS/RBS/PF.
    """
    df = flatten_headers(df.copy())

    alias: Dict[str, List[str]] = {
        "MIN":  ["MIN","Min","Minutes","Time","Min.","MINUTES"],
        "PTS":  ["PTS","Pts","Points","P","Total Points","Points Scored"],
        "2FG":  ["2P M/A","2P M/A %","2PM-A","2PT M/A","2-pt M/A","2PT","2P","2PTS","2PTs M/A","2 Points M/A","2pt M/A","2pt"],
        "3FG":  ["3P M/A","3P M/A %","3PM-A","3PT M/A","3-pt M/A","3PT","3P","3PTS","3PTs M/A","3 Points M/A","3pt M/A","3pt"],
        "FT":   ["FT M/A","FT M/A %","FTM-A","FTs M/A","Freethrows M/A","Free throws M/A","FT"],

        "O":    ["ORB","OReb","Off Reb","Offensive Rebounds","O"],
        "D":    ["DRB","DReb","Def Reb","Defensive Rebounds","D"],
        "T":    ["TRB","Tot Reb","REB","Total Rebounds","T","RBS"],  # RBS -> total rebounds

        "AST":  ["AST","Assists","A","AS"],    # AS (Greek short)
        "STL":  ["STL","Steals","S","ST"],     # ST (Greek short)
        "TOV":  ["TOV","TO","Turnovers","ΤΟ","TO."],  # TO (Greek short)

        "BLK_FV":["BLK","Blocks For","Blocks Made","Blocks","BS"],    # BS -> blocks made
        "BLK_AG":["BLK Against","Blocks Against","BA","Blocks Received"],

        "FLS_CM":["Fouls","Fouls Committed","PF","Cm","Personal Fouls","PF."],  # PF -> committed (συνήθως)
        "FLS_RV":["Fouls Drawn","FD","Rv","Fouls Received"],

        "PIR":  ["PIR","EFF","Index","Performance Index Rating","Efficiency"],

        "W":    ["W","Win"],
        "L":    ["L","Loss"],
        "Team": ["Team","Opponent","Opp."],
        "Team.1":["Team.1","Opponent.1","Opp.1"],
        "game_no":["Game","#","No","Match #","Α/Α"],
        "game_date":["Date","Ημερομηνία"],
        "opponent":["Opponent","Αντίπαλος","Team","Opp."],
    }

    # direct rename
    rename = {}
    for std, keys in alias.items():
        for c in list(df.columns):
            if c in keys:
                rename[c] = std
    df = df.rename(columns=rename)

    # fuzzy contains
    for std, keys in alias.items():
        if std in df.columns:
            continue
        for c in list(df.columns):
            lc = c.lower()
            if any(k.lower() in lc for k in keys):
                df = df.rename(columns={c: std})
                break

    # Διαχωρισμός made/attempted για 2FG/3FG/FT (x/y)
    def split_made_att(val):
        if isinstance(val, str) and "/" in val:
            a,b = val.split("/",1)
            try: return float(a), float(b)
            except: return a, b
        return None, None

    for cat in ("2FG","3FG","FT"):
        if cat in df.columns:
            m, a = zip(*[split_made_att(x) for x in df[cat].tolist()])
            df[f"{cat}_M"] = m
            df[f"{cat}_A"] = a

    # standard columns
    standard = [
        "MIN","PTS","2FG","3FG","FT","2FG_M","2FG_A","3FG_M","3FG_A","FT_M","FT_A",
        "O","D","T","AST","STL","TOV","BLK_FV","BLK_AG","FLS_CM","FLS_RV","PIR",
        "W","L","Team","Team.1","game_no","game_date","opponent"
    ]
    for col in standard:
        if col not in df.columns:
            df[col] = pd.NA

    # πέτα columns που είναι μόνο αριθμοί (όπως '1','2','3',…)
    junk_num_cols = [c for c in df.columns if re.fullmatch(r"\d+", str(c))]
    if junk_num_cols:
        df = df.drop(columns=junk_num_cols, errors="ignore")

    return df

# ---------------------- HTML parser ----------------------
def parse_bn_html_gamelogs(page_html: str) -> pd.DataFrame:
    tables = pd.read_html(page_html)
    if not tables:
        return pd.DataFrame()

    wanted_tokens = {"MIN","PTS","2P","3P","FT","REB","AST","STL","TO","EFF","PIR","AS","ST","BS","RBS","PF","Opponent","Date"}
    best = None
    best_score = -1
    for t in tables:
        t = flatten_headers(t)
        score = sum(1 for c in t.columns if any(tok in str(c).upper() for tok in wanted_tokens))
        if score > best_score or (score == best_score and t.shape[0] > (best.shape[0] if best is not None else 0)):
            best = t
            best_score = score

    if best is None or best.empty:
        return pd.DataFrame()

    df = best.copy()

    # drop συνοψιστικές (Total/Average)
    mask_sum_rows = df.iloc[:,0].astype(str).str.contains(r"Total|Average|Μέσος Όρος", case=False, na=False)
    df = df[~mask_sum_rows].reset_index(drop=True)

    # πρόσθεσε αύξοντα αν δεν υπάρχει
    if "game_no" not in df.columns:
        df.insert(0, "game_no", range(1, len(df)+1))

    df = unify_columns(df)

    # αν έχουμε Date/Opponent σε alternative labels, ανανέωσέ τα
    if "game_date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date":"game_date"})
    if "opponent" not in df.columns and "Opponent" in df.columns:
        df = df.rename(columns={"Opponent":"opponent"})

    return df

# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025")
    ap.add_argument("--competition", default="E")
    ap.add_argument("--mode", default="perGame")
    ap.add_argument("--player_code", default="", help="tag για τον παίκτη στο CSV")
    ap.add_argument("--player_name", default="", help="προαιρετικά, αν θες να γραφτεί ρητά")
    ap.add_argument("--json_url", default="", help="BN JSON endpoint (αν υπάρχει)")
    ap.add_argument("--page_url", default="", help="BN player page URL (HTML fallback)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.DataFrame()
    page_html = None

    # JSON πρώτα αν δοθεί
    if args.json_url:
        try:
            j = http_json(args.json_url)
            rows = None
            if isinstance(j, dict):
                for k in ("items","data","rows","list","games","matches"):
                    if isinstance(j.get(k), list):
                        rows = j[k]; break
            if rows is None and isinstance(j, list):
                rows = j
            if rows:
                df = pd.json_normalize(rows, sep="_")
            else:
                df = pd.json_normalize(j, sep="_")
            df = unify_columns(df)
            if "game_no" not in df.columns:
                df.insert(0, "game_no", range(1, len(df)+1))
        except Exception as e:
            print(f"[warn] JSON mode failed: {e}")

    # HTML fallback
    if df.empty and args.page_url:
        try:
            page_html = http_get(args.page_url)
            df = parse_bn_html_gamelogs(page_html)
        except Exception as e:
            print(f"[warn] HTML mode failed: {e}")

    if df.empty:
        print("No gamelog rows parsed.")
        sys.exit(0)

    if args.player_code:
        df["player_code"] = args.player_code

    # player_name: προτεραιότητα στο input, αλλιώς προσπάθησε από HTML
    pname = args.player_name.strip()
    if not pname and page_html:
        pname = extract_player_name_from_html(page_html) or ""
    if pname:
        df["player_name"] = pname

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
