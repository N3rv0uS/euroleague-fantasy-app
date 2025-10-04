\
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Κατεβάζει EuroLeague player stats ανά σεζόν και τα σώζει σε CSV/Excel/SQLite.
Προτιμά το πακέτο "euroleague-api". Αν δεν υπάρχει/αποτύχει, κάνει fallback σε raw HTTP στο public swagger (αν είναι διαθέσιμο).
"""

import argparse
import json
import os
import sys
from typing import List, Dict
import pandas as pd

# ---------------- Advanced Metrics ----------------
def _get_col(df, *candidates, default=0):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(default, index=df.index, dtype="float64")

def add_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Προσθέτει derived / advanced metrics χωρίς να ξαναυπολογίζει PIR.
    Απαιτεί όσο γίνεται τα παρακάτω (ονομασίες παίζουν ανά endpoint):
    - 2PM/2PA, 3PM/3PA ή FGM/FGA
    - FTM/FTA
    - PTS, AST, TOV, MIN, GP
    - team_name (ή Team)
    """

    out = df.copy()

    # Column aliases
    team = _get_col(out, "Team", "team_name")
    gp   = _get_col(out, "GP", "gp")
    min_ = _get_col(out, "MIN", "min")
    pts  = _get_col(out, "PTS", "pts")
    ast  = _get_col(out, "AST", "ast")
    tov  = _get_col(out, "TOV", "tov")
    oreb = _get_col(out, "OREB", "oreb")
    dreb = _get_col(out, "DREB", "dreb")
    treb = _get_col(out, "REB", "reb", default=oreb.add(dreb, fill_value=0))

    p2m  = _get_col(out, "2PM", "fg2m", "twoptm")
    p2a  = _get_col(out, "2PA", "fg2a", "twopta")
    p3m  = _get_col(out, "3PM", "fg3m", "threeptm")
    p3a  = _get_col(out, "3PA", "fg3a", "threepta")
    ftm  = _get_col(out, "FTM", "ftm")
    fta  = _get_col(out, "FTA", "fta")

    fgm  = _get_col(out, "FGM", "fgm", default=p2m + p3m)
    fga  = _get_col(out, "FGA", "fga", default=p2a + p3a)

    # ----- Shooting profiles -----
    # eFG% = (FGM + 0.5*3PM) / FGA
    out["eFG%"] = ((fgm + 0.5 * p3m) / fga).replace([pd.NA, pd.NaT], 0).fillna(0)

    # TS% = PTS / (2*(FGA + 0.44*FTA))
    denom_ts = (2 * (fga + 0.44 * fta))
    out["TS%"] = (pts / denom_ts).replace([pd.NA, pd.NaT], 0).fillna(0)

    # 3PAr = 3PA / FGA
    out["3PAr"] = (p3a / fga).replace([pd.NA, pd.NaT], 0).fillna(0)

    # FTr = FTA / FGA
    out["FTr"] = (fta / fga).replace([pd.NA, pd.NaT], 0).fillna(0)

    # AST/TOV
    out["AST/TOV"] = (ast / tov.replace(0, pd.NA)).fillna(pd.NA).replace([pd.NaT], pd.NA)
    out["AST/TOV"] = out["AST/TOV"].fillna(0)

    # TOV% ≈ TOV / (FGA + 0.44*FTA + TOV)
    denom_tov = (fga + 0.44*fta + tov)
    out["TOV%"] = (tov / denom_tov).replace([pd.NA, pd.NaT], 0).fillna(0)

    # ----- Per-36 pace-neutral rates -----
    # Προτιμούμε per-36 για συγκρισιμότητα.
    min_safe = min_.replace(0, pd.NA)
    for col, label in [(pts, "PTS/36"), (treb, "REB/36"), (ast, "AST/36"),
                       (tov, "TOV/36"), (p3m, "3PM/36"), (oreb, "OREB/36"),
                       (dreb, "DREB/36")]:
        out[label] = (col / min_safe * 36).fillna(0)

    # ----- Usage% (προσεγγιστικό, από player+team totals) -----
    # USG% = 100 * ((FGA + 0.44*FTA + TOV) * (TeamMinutes/5)) / (MIN * (TeamFGA + 0.44*TeamFTA + TeamTOV))
    # Θα προσεγγίσουμε TeamMinutes/παιχνίδι από τα αθροίσματα των παικτών. Αν λείπει -> 200.
    # Για TeamFGA/FTA/TOV ανά παιχνίδι: άθροισμα per-game παικτών της ίδιας ομάδας.
    # Group by team:
    out["_player_poss_comp"] = (fga + 0.44*fta + tov)
    # Χρειαζόμαστε ένα καθαρό string με team id για groupby
    team_key = team.astype(str).fillna("Unknown")
    agg = out.groupby(team_key).agg(
        Team_MIN_pg = ("MIN", "sum") if "MIN" in out.columns else ("min", "sum"),
        Team_FGA_pg = (lambda x: fga.groupby(team_key).sum()[x.name]) if False else ("_player_poss_comp", "sum"), # placeholder
    )

    # Επειδή δεν μπορούμε να χρησιμοποιήσουμε custom στο ίδιο .agg για διαφορετικές στήλες εύκολα, ξαναφτιάχνουμε:
    team_df = pd.DataFrame({
        "Team": team_key,
        "MIN": min_,
        "FGA": fga,
        "FTA": fta,
        "TOV": tov,
    })
    team_totals = team_df.groupby("Team").sum(numeric_only=True)
    team_totals["TeamMinutes_pg"] = team_totals["MIN"].where(team_totals["MIN"] > 0, 200)
    team_totals["TeamPossComp_pg"] = team_totals["FGA"] + 0.44*team_totals["FTA"] + team_totals["TOV"]
    team_totals = team_totals[["TeamMinutes_pg", "TeamPossComp_pg", "FGA", "FTA", "TOV"]]

    out = out.join(team_totals, on=team_key, how="left")

    denom_usg = (min_safe * out["TeamPossComp_pg"])
    numer_usg = (out["_player_poss_comp"] * (out["TeamMinutes_pg"] / 5.0))
    out["USG%"] = (100 * numer_usg / denom_usg).replace([pd.NA, pd.NaT], 0).fillna(0)

    # ----- AST% (playmaking share, simplified) -----
    # AST% ≈ 100 * AST / ( (MIN/TeamMinutes)*TeamFGM - FGM )
    team_fgm = team_df.groupby("Team")["FGA"].sum() * 0  # placeholder to keep index
    # Αν έχουμε FGM στήλη, πάρτην από team_df (δεν την έχουμε εδώ). Θα υπολογίσουμε ως 2PM+3PM αν λείπει.
    team_fgm_df = pd.DataFrame({
        "Team": team_key,
        "FGM": fgm
    }).groupby("Team").sum(numeric_only=True)
    out = out.join(team_fgm_df.rename(columns={"FGM":"TeamFGM_pg"}), on=team_key, how="left")

    denom_astp = ((min_safe / out["TeamMinutes_pg"]) * out["TeamFGM_pg"] - fgm)
    out["AST%"] = (100 * ast / denom_astp.replace(0, pd.NA)).fillna(0)

    # Καθαρισμοί βοηθητικών
    out.drop(columns=[c for c in ["_player_poss_comp"] if c in out.columns], inplace=True)

    # Clip σε λογικά όρια
    for c in ["eFG%", "TS%", "3PAr", "FTr", "TOV%"]:
        out[c] = out[c].clip(lower=0, upper=1)
    for c in ["USG%", "AST%"]:
        out[c] = out[c].clip(lower=0, upper=100)

    return out

# Προαιρετικά: χρήση του community package
def try_import_euroleague_api():
    try:
        from euroleague_api.player_stats import PlayerStats  # type: ignore
        from euroleague_api.euroleague_data import EuroleagueData  # type: ignore
        return PlayerStats, EuroleagueData
    except Exception:
        return None, None

def fetch_with_package(season: int, competition_code: str, mode: str) -> pd.DataFrame:
    PlayerStats, EuroleagueData = try_import_euroleague_api()
    if PlayerStats is None:
        raise RuntimeError("Το package 'euroleague-api' δεν είναι διαθέσιμο.")
    # Τα ονόματα/μέθοδοι βασίζονται στη βιβλιοθήκη (βλ. README του project).
    ps = PlayerStats(competition_code=competition_code)
    # Συνήθη modes: perGame | perMinute | accumulated
    df = ps.get_season_player_stats(season=season, statistic_mode=mode)
    # Εμπλουτισμός με metadata ομάδων/παικτών αν διαθέσιμα
    return df

def fetch_with_raw_requests(season: int, competition_code: str, mode: str) -> pd.DataFrame:
    """
    Χρησιμοποιεί το ίδιο endpoint με το site (stats/expanded).
    """
    import requests, pandas as pd

    base = "https://www.euroleaguebasketball.net/el/euroleague/stats/expanded/"
    season_code = f"{competition_code}{season}"  # π.χ. E2025

    url = (
        f"{base}?size=1000&viewType=traditional"
        f"&seasonMode=Range"
        f"&statisticMode={mode}"
        f"&fromSeasonCode={season_code}&toSeasonCode={season_code}"
        f"&sortDirection=ascending&statistic="
    )

    print("🔎 Trying URL:", url)

    r = requests.get(url, timeout=60)
    print("🔎 Status code:", r.status_code)
    print("🔎 Response snippet:", r.text[:500])  # δείξε πρώτα 500 χαρακτήρες

    # Πολλές φορές το response είναι JSON (αν το καλέσεις απευθείας με headers)
    try:
        data = r.json()
        rows = data.get("data", data)
        df = pd.json_normalize(rows)
        if len(df) > 0:
            return df
    except Exception as e:
        print("⚠️ JSON decode failed:", e)

    # Αν δεν είναι JSON, γύρισε empty DataFrame
    return pd.DataFrame()



def write_outputs(df: pd.DataFrame, season: int, mode: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"players_{season}_{mode}.csv")
    xlsx_path = os.path.join(out_dir, f"players_{season}_{mode}.xlsx")
    db_path = os.path.join(out_dir, "euroleague.db")

    # Κανονικοποίηση βασικών aliases στηλών (αν υπάρχουν)
    colmap = {
        "player_name": "Player",
        "team_name": "Team",
        "pts": "PTS",
        "reb": "REB",
        "ast": "AST",
        "stl": "STL",
        "blk": "BLK",
        "min": "MIN",
        "fg3m": "3PM",
        "fg3a": "3PA",
        "fg3pct": "3P%",
        "fg2m": "2PM",
        "fg2a": "2PA",
        "fg2pct": "2P%",
        "ftm": "FTM",
        "fta": "FTA",
        "ftpct": "FT%",
        "oreb": "OREB",
        "dreb": "DREB",
        "tov": "TOV",
        "pf": "PF",
        "gp": "GP",
        "gs": "GS",
    }
    for k, v in colmap.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception:
        pass

    # SQLite
    import sqlite3
    con = sqlite3.connect(db_path)
    df.to_sql("player_stats", con, if_exists="replace", index=False)
    con.close()

    return csv_path, xlsx_path, db_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, help="Π.χ. 2025 2024", required=False)
    parser.add_argument("--mode", type=str, default="perGame", choices=["perGame", "perMinute", "accumulated"])
    parser.add_argument("--competition", type=str, default="E", help="E=EuroLeague, U=EuroCup")
    parser.add_argument("--out", type=str, default="out", help="Φάκελος εξόδου")
    parser.add_argument("--force-raw", action="store_true", help="Παράκαμψη βιβλιοθήκης και χρήση raw HTTP")
    args = parser.parse_args()

    # Φόρτωσε config αν δεν δοθούν seasons
    cfg_path = "config.json"
    if args.seasons is None and os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        seasons = cfg.get("seasons", [2025])
        mode = cfg.get("statistic_mode", args.mode)
        competition = cfg.get("competition_code", args.competition)
        out_dir = cfg.get("output_dir", args.out)
    else:
        seasons = args.seasons or [2025]
        mode = args.mode
        competition = args.competition
        out_dir = args.out

    used_raw = False
    for season in seasons:
        try:
            if args.force_raw:
                used_raw = True
                df = fetch_with_raw_requests(season, competition, mode)
            else:
                try:
                    df = fetch_with_package(season, competition, mode)
                except Exception:
                    used_raw = True
                    df = fetch_with_raw_requests(season, competition, mode)
        except Exception as e:
            print(f"[Σφάλμα] Season {season}: {e}", file=sys.stderr)
            continue

        # Advanced metrics enrichment
        try:
            df = add_advanced_metrics(df)
        except Exception as _e:
            print(f"[Προειδοποίηση] Advanced metrics: {_e}")
        csv_path, xlsx_path, db_path = write_outputs(df, season, mode, out_dir)
        print(f"✔ Season {season}:")
        print(f"  - CSV:  {csv_path}")
        print(f"  - XLSX: {xlsx_path}")
        print(f"  - DB:   {db_path}")
    if used_raw:
        print("\n(Χρησιμοποιήθηκε fallback με raw HTTP — αν κάτι δεν κατέβηκε, έλεγξε τα endpoints/headers.)")

if __name__ == "__main__":
    main()
