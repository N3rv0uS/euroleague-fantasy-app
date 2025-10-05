#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, time, json
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0"}
BASE = "https://feeds.incrowdsports.com/provider/euroleague-feeds/v3/competitions"

# Παραλλαγές για brute-force:
PLAYER_KEYS   = ["playerCodes", "playerCode", "playerIds", "code"]
PHASE_KEYS    = [None, "RS", "PO"]                  # Regular Season / Playoffs
SPLIT_MODES   = ["ByGame"]                          # αυτό θέλουμε για gamelogs
SORT_MODES    = ["GameDate", "GameDateDesc"]        # για κάθε ενδεχόμενο

def get_json(url: str, params: Dict[str, Any], timeout: int = 25) -> Tuple[int, Optional[dict], Optional[str]]:
    try:
        r = requests.get(url, params=params, headers=UA, timeout=timeout)
        status = r.status_code
        if status != 200:
            return status, None, f"http {status}"
        try:
            return status, r.json(), None
        except Exception as e:
            return status, None, f"json error: {e}"
    except Exception as e:
        return 0, None, f"req error: {e}"

def probe_one_player(competition: str, season_code: str, mode: str, player_code: str) -> pd.DataFrame:
    """
    Δοκιμάζει παραλλαγές παραμέτρων μέχρι να επιστρέψει rows.
    Επιστρέφει DataFrame (ή empty DF).
    """
    url = f"{BASE}/{competition}/statistics/players/traditional"
    common = dict(
        seasonMode="Range",
        fromSeasonCode=season_code,
        toSeasonCode=season_code,
        statisticMode=mode,
        limit=1000,
    )

    trials = []
    for split in SPLIT_MODES:
        for sort in SORT_MODES:
            for pkey in PLAYER_KEYS:
                for phase in PHASE_KEYS:
                    params = common.copy()
                    params["statisticSplitMode"] = split
                    params["statisticSortMode"]  = sort
                    params[pkey] = player_code
                    if phase:
                        # Δουλεύουν και gamePhaseCode ή gamePhaseType σε άλλα feeds – δοκιμάζουμε.
                        params["gamePhaseCode"] = phase
                        # params["gamePhaseType"] = phase  # αν χρειαστεί, un-comment

                    trials.append(params)

    # εκτέλεση
    for params in trials:
        status, js, err = get_json(url, params)
        if js and isinstance(js, dict):
            # Συνήθως data -> rows, αλλά το schema αλλάζει· ψάχνουμε λογικές λίστες
            rows = None
            for k in ("rows","items","data","list"):
                if isinstance(js.get(k), list):
                    rows = js[k]
                    break
            # Μερικές φορές επιστρέφει {errorMessage: ...}
            if not rows and any(key in js for key in ("errorMessage","propertyName","message")):
                continue
            if not rows and isinstance(js, list):
                rows = js

            if rows:
                df = pd.json_normalize(rows, sep="_")
                # πεδία που συνήθως υπάρχουν:
                # gameDate / fixtureDate, opponentTeamCode, teamCode, player_code κ.λπ.
                # Ομογενοποίηση βασικών:
                alias = {
                    "player.code": "player_code",
                    "playerCode": "player_code",
                    "player_code": "player_code",
                    "player.name": "player_name",
                    "playerName": "player_name",
                    "team.code": "team_code",
                    "teamCode": "team_code",
                    "opponentTeamCode": "opponent_code",
                    "opponent.code": "opponent_code",
                    "gameDate": "game_date",
                    "fixtureDate": "game_date",
                }
                for old, new in alias.items():
                    if old in df.columns and new not in df.columns:
                        df = df.rename(columns={old:new})

                # Προσθέτουμε μεταδεδομένα
                df["season"] = season_code.replace("E","").replace("U","")
                df["competition"] = competition
                df["mode"] = mode

                # Basic columns we care about: MIN, PTS, REB(T), AST, STL, TOV, BLK_FV, FLS_CM, PIR
                # Σε άλλα feeds μπορεί να ονομάζονται διαφορετικά (π.χ. totalRebounds, assists κ.λπ.)
                rename_stats = {
                    "minutesPlayed": "MIN",
                    "pointsScored": "PTS",
                    "totalRebounds": "T",
                    "assists": "AST",
                    "steals": "STL",
                    "turnovers": "TOV",
                    "blocks": "BLK_FV",
                    "foulsCommited": "FLS_CM",   # incrowd typo συχνό
                    "foulsCommitted": "FLS_CM",
                    "pir": "PIR",
                    "twoPointersMade": "2FG_M",
                    "twoPointersAttempted": "2FG_A",
                    "threePointersMade": "3FG_M",
                    "threePointersAttempted": "3FG_A",
                    "freeThrowsMade": "FT_M",
                    "freeThrowsAttempted": "FT_A",
                }
                for old, new in rename_stats.items():
                    if old in df.columns and new not in df.columns:
                        df = df.rename(columns={old:new})

                # ταξινόμηση κατά ημερομηνία αν υπάρχει
                if "game_date" in df.columns:
                    with pd.option_context('mode.use_inf_as_na', True):
                        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
                    df = df.sort_values("game_date").reset_index(drop=True)

                return df

    # τίποτα
    return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, help="π.χ. 2025")
    ap.add_argument("--competition", default="E")
    ap.add_argument("--mode", default="perGame")
    ap.add_argument("--players_csv", default="", help="default: out/players_<season>_<mode>.csv")
    ap.add_argument("--players", default="", help="comma list για γρήγορο smoke (π.χ. 002661,011196)")
    ap.add_argument("--out_dir", default="out")
    args = ap.parse_args()

    season_code = f"{args.competition}{args.season}"
    os.makedirs(args.out_dir, exist_ok=True)

    if args.players:
        player_list = [x.strip() for x in args.players.split(",") if x.strip()]
    else:
        players_csv = args.players_csv or os.path.join(args.out_dir, f"players_{args.season}_{args.mode}.csv")
        if not os.path.exists(players_csv):
            print(f"Δεν βρέθηκε {players_csv}")
            sys.exit(1)
        pdf = pd.read_csv(players_csv)
        # Πεδίο player_code στο season feed
        if "player_code" in pdf.columns:
            player_list = pdf["player_code"].astype(str).unique().tolist()
        elif "code" in pdf.columns:
            player_list = pdf["code"].astype(str).unique().tolist()
        else:
            print("Δεν βρέθηκε στήλη player_code/code στο players CSV.")
            sys.exit(1)

    all_rows: List[pd.DataFrame] = []
    for i, pcode in enumerate(player_list, start=1):
        df = probe_one_player(args.competition, season_code, args.mode, pcode)
        if not df.empty:
            # γράψε ποιος παίκτης είναι (αν δεν υπάρχει)
            if "player_code" not in df.columns:
                df["player_code"] = pcode
            all_rows.append(df)
            print(f"[{i}/{len(player_list)}] {pcode}: +{len(df)} rows")
        else:
            print(f"[{i}/{len(player_list)}] {pcode}: 0 rows")

        # μικρή καθυστέρηση να μην “γκρινιάζει” το feed
        time.sleep(0.15)

    if not all_rows:
        print("No gamelogs retrieved.")
        sys.exit(0)

    out = pd.concat(all_rows, ignore_index=True)
    out_csv = os.path.join(args.out_dir, f"player_gamelogs_{args.season}_{args.mode}.csv")
    out.to_csv(out_csv, index=False)
    print(f"[OK] saved {out_csv} rows={len(out)}")

if __name__ == "__main__":
    main()
