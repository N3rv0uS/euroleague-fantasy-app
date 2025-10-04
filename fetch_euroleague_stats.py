#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Κατεβάζει EuroLeague player stats ανά σεζόν και τα σώζει σε CSV/Excel/SQLite.
1) Προσπαθεί API (api-live.euroleague.net) με seasonMode=Range / seasonCode / season
2) Fallback: κάνει scrape τον πίνακα από το /euroleague/stats/expanded/
Περιλαμβάνει basic normalization (Ελληνικά headers -> Αγγλικά) και advanced metrics.
"""

import argparse
import json
import os
import sys
from typing import List, Dict
import pandas as pd
import time
import requests


INCROWD_BASE = "https://feeds.incrowdsports.com/provider/euroleague-feeds/v3"

def _norm_pct(series):
    if series.dtype == object:
        s = (series.astype(str)
                  .str.replace("%", "", regex=False)
                  .str.replace(",", ".", regex=False))
        return pd.to_numeric(s, errors="coerce")
    return series

def fetch_player_games(player_code: str, season: int,
                       competition: str = "E",
                       mode: str = "perGame",
                       limit: int = 1000) -> pd.DataFrame:
    """
    Gamelogs ενός παίκτη (ένα row ανά παιχνίδι) από IncrowdSports JSON feed.
    - player_code: π.χ. '002661'
    - season: 2025 (χωρίς Ε)
    - competition: 'E' (EuroLeague) ή 'U' (EuroCup)
    - mode: 'perGame' | 'perMinute' | 'accumulated'
    """
    season_code = f"{competition}{season}"
    url = (
        f"{INCROWD_BASE}/competitions/{competition}/statistics/players/traditional"
        f"?seasonMode=Range&fromSeasonCode={season_code}&toSeasonCode={season_code}"
        f"&statisticMode={mode}&statisticSortMode=GameDate"
        f"&playerCodes={player_code}&limit={limit}"
    )
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Το response έχει συνήθως key "players" με λίστα παικτών, όπου κάθε παίκτης έχει "games"
    players = data.get("players", [])
    if not players:
        return pd.DataFrame()

    # συνήθως είναι ένας παίκτης – παίρνουμε όλα τα παιχνίδια
    rows = []
    for p in players:
        games = p.get("games", [])
        if not games:
            continue
        df = pd.json_normalize(games)
        # εμπλουτισμός χρήσιμων metadata του παίκτη
        meta = {
            "player.code": p.get("player", {}).get("code"),
            "player.name": p.get("player", {}).get("name"),
            "team.code": p.get("team", {}).get("tvCodes"),
            "team.name": p.get("team", {}).get("name"),
        }
        for k, v in meta.items():
            df[k] = v
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    # ονοματοδοσία βασικών στηλών (όπου υπάρχουν)
    rename = {
        "gameCode": "GameCode",
        "roundNumber": "Round",
        "date": "GameDate",
        "homeOrAway": "HA",  # 'H' ή 'A'
        "opponent.tvCodes": "OppCode",
        "opponent.name": "OppName",
        "minutesPlayed": "MIN",
        "pointsScored": "PTS",
        "assists": "AST",
        "turnovers": "TOV",
        "offensiveRebounds": "OREB",
        "defensiveRebounds": "DREB",
        "totalRebounds": "REB",
        "steals": "STL",
        "blocks": "BLK",
        "foulsCommitted": "PF",
        "pir": "PIR",
        "twoPointersMade": "2PM",
        "twoPointersAttempted": "2PA",
        "twoPointersPercentage": "2P%",
        "threePointersMade": "3PM",
        "threePointersAttempted": "3PA",
        "threePointersPercentage": "3P%",
        "freeThrowsMade": "FTM",
        "freeThrowsAttempted": "FTA",
        "freeThrowsPercentage": "FT%",
    }
    out.rename(columns={k: v for k, v in rename.items() if k in out.columns}, inplace=True)

    # καθάρισε ποσοστά
    for col in ["2P%", "3P%", "FT%"]:
        if col in out.columns:
            out[col] = _norm_pct(out[col])

    # Μετατροπές τύπων
    # GameDate σε datetime (αν υπάρχει)
    if "GameDate" in out.columns:
        out["GameDate"] = pd.to_datetime(out["GameDate"], errors="coerce")

    return out


def fetch_all_player_gamelogs(season: int,
                              competition: str = "E",
                              mode: str = "perGame",
                              rate_sleep: float = 0.2) -> pd.DataFrame:
    """
    Φέρνει gamelogs για ΟΛΟΥΣ τους παίκτες της σεζόν που υπάρχουν στο master feed.
    """
    season_code = f"{competition}{season}"
    # master feed για να πάρουμε όλους τους player codes
    master_url = (
        f"{INCROWD_BASE}/competitions/{competition}/statistics/players/traditional"
        f"?seasonMode=Range&limit=1000&sortDirection=ascending"
        f"&fromSeasonCode={season_code}&toSeasonCode={season_code}"
        f"&statisticMode={mode}"
    )
    players = requests.get(master_url, timeout=60).json().get("players", [])
    master_df = pd.json_normalize(players)
    codes = master_df.get("player.code")
    if codes is None or master_df.empty:
        return pd.DataFrame()

    codes = master_df["player.code"].dropna().astype(str).unique().tolist()

    all_rows = []
    for code in codes:
        try:
            df = fetch_player_games(code, season, competition, mode)
            if len(df):
                all_rows.append(df)
        except Exception:
            # αν κάποιος παίκτης αποτύχει, συνέχισε
            pass
        time.sleep(rate_sleep)  # ευγένεια προς το feed

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)

# ---------------- Helpers & Advanced Metrics ----------------
def _get_col(df, *candidates, default=0):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(default, index=df.index, dtype="float64")

def fetch_from_incrowd(season: int, competition_code: str, mode: str) -> pd.DataFrame:
    """
    Τραβάει JSON από το επίσημο feeds.incrowdsports.com (EuroLeague feed).
    competition_code: 'E' για EuroLeague, 'U' για EuroCup (αν χρειαστεί).
    mode: perGame | perMinute | accumulated
    """
    import requests, pandas as pd

    base = "https://feeds.incrowdsports.com/provider/euroleague-feeds/v3/competitions"
    season_code = f"{competition_code}{season}"  # π.χ. E2025

    url = (
        f"{base}/{competition_code}/statistics/players/traditional"
        f"?seasonMode=Range&limit=1000&sortDirection=ascending"
        f"&fromSeasonCode={season_code}&toSeasonCode={season_code}"
        f"&statisticMode={mode}&statisticSortMode={mode}"
    )

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Συνήθως τα rows είναι στη ρίζα σε key "players" ή παρόμοιο
    rows = data.get("players", data)
    df = pd.json_normalize(rows)

    # Μικρή ονοματοδοσία για βασικά πεδία που βλέπω στο response σου
    rename = {
        "player.name": "Player",
        "team.name": "Team",
        "team.tvCodes": "TeamCode",
        "minutesPlayed": "MIN",
        "pointsScored": "PTS",
        "assists": "AST",
        "turnovers": "TOV",
        "defensiveRebounds": "DREB",
        "offensiveRebounds": "OREB",
        "totalRebounds": "REB",
        "threePointersMade": "3PM",
        "threePointersAttempted": "3PA",
        "twoPointersMade": "2PM",
        "twoPointersAttempted": "2PA",
        "freeThrowsMade": "FTM",
        "freeThrowsAttempted": "FTA",
        "foulsCommitted": "PF",
        "steals": "STL",
        "blocks": "BLK",
        "gamesPlayed": "GP",
        "gamesStarted": "GS",
        "pir": "PIR",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    # Μετατρέπουμε ό,τι είναι ποσοστό σε αριθμό (χωρίς %)
    for pct_col in [
        "twoPointersPercentage", "threePointersPercentage", "freeThrowsPercentage"
    ]:
        if pct_col in df.columns:
            df[pct_col] = (
                df[pct_col].astype(str).str.replace("%", "", regex=False)
            )
            df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce")

    # Minutes πιθανόν έρχονται ήδη ως δεκαδικά (π.χ. 6.8333) – κρατάμε όπως είναι
    return df

def _fetch_table_with_playwright(url: str) -> pd.DataFrame:
    """
    Ανοίγει headless Chromium, φορτώνει τη σελίδα, περιμένει να εμφανιστεί ο πίνακας
    και επιστρέφει DataFrame από το πρώτο <table>.
    """
    from playwright.sync_api import sync_playwright
    import pandas as pd
    from bs4 import BeautifulSoup

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            locale="en-US"
        )
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)

        # περιμένουμε να φορτώσει ο πίνακας (δοκίμασε για <table>)
        try:
            page.wait_for_selector("table", timeout=120_000)
        except Exception:
            # μερικές φορές αργεί – κάνε λίγο scroll/αναμονή
            page.wait_for_timeout(3000)

        html = page.content()
        browser.close()

    # Ανάλυση HTML με BeautifulSoup και μετατροπή σε DF
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None:
        soup = BeautifulSoup(html, "html5lib")
        table = soup.find("table")
    if table is None:
        return pd.DataFrame()

    df = pd.read_html(str(table))[0]
    return df

def add_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Προσθέτει derived / advanced metrics (χωρίς ξανα-υπολογισμό PIR).
    Χρησιμοποιεί aliases για να “βρει” γνωστές στήλες όπου υπάρχουν.
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

    # Shooting profiles
    out["eFG%"] = ((fgm + 0.5 * p3m) / fga).replace([pd.NA, pd.NaT], 0).fillna(0)
    denom_ts = (2 * (fga + 0.44 * fta))
    out["TS%"] = (pts / denom_ts).replace([pd.NA, pd.NaT], 0).fillna(0)
    out["3PAr"] = (p3a / fga).replace([pd.NA, pd.NaT], 0).fillna(0)
    out["FTr"]  = (fta / fga).replace([pd.NA, pd.NaT], 0).fillna(0)

    out["AST/TOV"] = (ast / tov.replace(0, pd.NA)).fillna(0)
    denom_tov = (fga + 0.44*fta + tov)
    out["TOV%"] = (tov / denom_tov).replace([pd.NA, pd.NaT], 0).fillna(0)

    # Per-36 pace-neutral rates
    min_safe = min_.replace(0, pd.NA) if "MIN" in out.columns else pd.Series(pd.NA, index=out.index)
    for col, label in [(pts, "PTS/36"), (treb, "REB/36"), (ast, "AST/36"),
                       (tov, "TOV/36"), (p3m, "3PM/36"), (oreb, "OREB/36"),
                       (dreb, "DREB/36")]:
        out[label] = (col / min_safe * 36).fillna(0)

    # Usage% (προσεγγιστικό)
    out["_player_poss_comp"] = (_get_col(out, "FGA", "fga", default=p2a + p3a)
                                + 0.44*_get_col(out, "FTA", "fta")
                                + _get_col(out, "TOV", "tov"))
    team_key = _get_col(out, "Team", "team_name").astype(str).fillna("Unknown")
    team_df = pd.DataFrame({
        "Team": team_key,
        "MIN": min_,
        "FGA": _get_col(out, "FGA", "fga", default=p2a + p3a),
        "FTA": _get_col(out, "FTA", "fta"),
        "TOV": _get_col(out, "TOV", "tov"),
        "FGM": _get_col(out, "FGM", "fgm", default=p2m + p3m),
    })
    team_totals = team_df.groupby("Team").sum(numeric_only=True)
    team_totals["TeamMinutes_pg"]   = team_totals["MIN"].where(team_totals["MIN"] > 0, 200)
    team_totals["TeamPossComp_pg"]  = team_totals["FGA"] + 0.44*team_totals["FTA"] + team_totals["TOV"]
    team_totals = team_totals[["TeamMinutes_pg", "TeamPossComp_pg", "FGM"]]
    out = out.join(team_totals.rename(columns={"FGM": "TeamFGM_pg"}), on=team_key, how="left")

    denom_usg = (min_safe * out["TeamPossComp_pg"])
    numer_usg = (out["_player_poss_comp"] * (out["TeamMinutes_pg"] / 5.0))
    out["USG%"] = (100 * numer_usg / denom_usg).replace([pd.NA, pd.NaT], 0).fillna(0)

    # AST%
    team_fgm = out["TeamFGM_pg"]
    denom_astp = ((min_safe / out["TeamMinutes_pg"]) * team_fgm - _get_col(out, "FGM", "fgm", default=p2m + p3m))
    out["AST%"] = (100 * ast / denom_astp.replace(0, pd.NA)).fillna(0)

    # Cleanups / bounds
    for c in ["eFG%", "TS%", "3PAr", "FTr", "TOV%"]:
        out[c] = out[c].clip(lower=0, upper=1)
    for c in ["USG%", "AST%"]:
        out[c] = out[c].clip(lower=0, upper=100)
    for c in ["_player_poss_comp"]:
        if c in out.columns:
            out.drop(columns=[c], inplace=True)

    return out


# ---------------- Optional community package ----------------
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
    ps = PlayerStats(competition_code=competition_code)
    df = ps.get_season_player_stats(season=season, statistic_mode=mode)
    return df


# ---------------- HTTP API + Scrape fallback ----------------
def fetch_with_raw_requests(season: int, competition_code: str, mode: str) -> pd.DataFrame:
    """
    1) Προσπαθεί σύγχρονα API endpoints του api-live.euroleague.net
    2) Fallback: κάνει scraping τον πίνακα από το /stats/expanded/ (όπως στο site)
    """
    import re
    import requests

    base_api = "https://api-live.euroleague.net"
    base_web = "https://www.euroleaguebasketball.net/el/euroleague/stats/expanded/"
    season_code = f"{competition_code}{season}"  # π.χ. E2025

    # (A) API candidates
    api_candidates = [
        f"{base_api}/v1/players/stats?seasonMode=Range&fromSeasonCode={season_code}"
        f"&toSeasonCode={season_code}&competitionCode={competition_code}&statisticMode={mode}&size=10000",
        f"{base_api}/v1/players/stats?seasonCode={season_code}&competitionCode={competition_code}"
        f"&statisticMode={mode}&size=10000",
        f"{base_api}/v1/players/stats?season={season}&competitionCode={competition_code}&statisticMode={mode}",
    ]

    last_err = None
    for url in api_candidates:
        try:
            print(f"🔎 Trying API URL: {url}")
            r = requests.get(url, timeout=60)
            print("   → status:", r.status_code)
            r.raise_for_status()
            data = r.json()
            rows = data.get("data", data)
            df = pd.json_normalize(rows)
            if len(df) > 0:
                print(f"✅ API returned {len(df)} rows")
                return df
        except Exception as e:
            last_err = e
            print(f"⚠️ API attempt failed: {e}")

        # ---------- (B) Fallback: scrape από expanded page ----------
    base_web = "https://www.euroleaguebasketball.net/en/euroleague/stats/expanded/"
    web_url = (
        f"{base_web}?size=1000&viewType=traditional"
        f"&seasonMode=Range&statisticMode={mode}"
        f"&fromSeasonCode={season_code}&toSeasonCode={season_code}"
        f"&sortDirection=ascending&statistic="
    )
    try:
        print(f"🔎 Fallback to HTML table: {web_url}")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.euroleaguebasketball.net/",
        }
        r = requests.get(web_url, headers=headers, timeout=60)
        r.raise_for_status()
        html = r.text
        print("   → downloaded HTML length:", len(html))

        # 1η προσπάθεια: read_html πάνω στο HTML
        try:
            tables = pd.read_html(html, flavor="lxml")
        except Exception:
            tables = pd.read_html(html, flavor="html5lib")

        if tables:
            df = tables[0].copy()
            print(f"✅ read_html found table: {df.shape}")
        else:
            print("ℹ️ No table via read_html — trying Playwright headless browser…")
            df = _fetch_table_with_playwright(web_url)
            if df is None or df.empty:
                print("⚠️ Playwright also found no table.")
                return pd.DataFrame()
            print(f"✅ Playwright found table: {df.shape}")

        # --- normalize όπως πριν ---
        import re
        df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
        if df.columns and df.columns[0].strip().lower() in {"#", "unnamed: 0"}:
            df = df.iloc[:, 1:]

        if "MIN" in df.columns and df["MIN"].dtype == object:
            def _mmss_to_min(x):
                s = str(x).strip()
                if ":" in s:
                    parts = s.split(":")
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        return int(parts[0]) + int(parts[1]) / 60.0
                try:
                    return float(s.replace(",", "."))
                except:
                    return None
            df["MIN"] = df["MIN"].apply(_mmss_to_min)

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col].astype(str)
                    .str.replace("%", "", regex=False)
                    .str.replace(",", ".", regex=False)
                )
                df[col] = pd.to_numeric(df[col], errors="ignore")

        for need in ["Player", "Team", "GP", "MIN", "PTS", "AST", "TOV", "REB"]:
            if need not in df.columns:
                df[need] = pd.NA

        return df

    except Exception as e:
        print(f"❌ HTML scrape failed: {e}")
        return pd.DataFrame()




# ---------------- Write outputs ----------------
def write_outputs(df: pd.DataFrame, season: int, mode: str, out_dir: str):
    if df is None or len(df) == 0:
        print(f"⚠️ Empty DataFrame for season {season} — skipping write.")
        return None, None, None

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"players_{season}_{mode}.csv")
    xlsx_path = os.path.join(out_dir, f"players_{season}_{mode}.xlsx")
    db_path  = os.path.join(out_dir, "euroleague.db")

    # Κανονικοποίηση aliases βασικών στηλών
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

    # SQLite (μόνο αν υπάρχουν στήλες)
    try:
        import sqlite3
        con = sqlite3.connect(db_path)
        df.to_sql("player_stats", con, if_exists="replace", index=False)
        con.close()
    except Exception as e:
        print(f"⚠️ SQLite write skipped: {e}")

    return csv_path, xlsx_path, db_path


# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, help="Π.χ. 2025 2024", required=False)
    parser.add_argument("--mode", type=str, default="perGame", choices=["perGame", "perMinute", "accumulated"])
    parser.add_argument("--competition", type=str, default="E", help="E=EuroLeague, U=EuroCup")
    parser.add_argument("--out", type=str, default="out", help="Φάκελος εξόδου")
    parser.add_argument("--force-raw", action="store_true", help="Παράκαμψη βιβλιοθήκης και χρήση raw HTTP/scrape")
    parser.add_argument("--kind", type=str, default="season",
                    choices=["season", "gamelogs"],
                    help="season: season averages | gamelogs: per-game stats")
    parser.add_argument("--players", type=str, default="",
                    help="Comma-separated player codes (π.χ. 002661,011196). Κενό = όλοι.")

    args = parser.parse_args()

    # Φόρτωσε config αν δεν δοθούν seasons
    cfg_path = "config.json"
    if args.seasons is None and os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        seasons     = cfg.get("seasons", [2025])
        mode        = cfg.get("statistic_mode", args.mode)
        competition = cfg.get("competition_code", args.competition)
        out_dir     = cfg.get("output_dir", args.out)
    else:
        seasons     = args.seasons or [2025]
        mode        = args.mode
        competition = args.competition
        out_dir     = args.out

    used_raw = False
  for season in seasons:
    try:
        if args.kind == "gamelogs":
            if args.players.strip():
                # συγκεκριμένοι παίκτες
                codes = [c.strip() for c in args.players.split(",") if c.strip()]
                frames = []
                for code in codes:
                    frames.append(fetch_player_games(code, season, competition, mode))
                df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                out_name = f"player_gamelogs_{season}_{mode}.csv"
            else:
                # όλοι οι παίκτες
                df = fetch_all_player_gamelogs(season, competition, mode)
                out_name = f"player_gamelogs_{season}_{mode}.csv"

            if df is None or df.empty:
                print(f"[Προειδοποίηση] Κενό gamelogs για season {season}")
                continue

            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, out_name)
            df.to_csv(csv_path, index=False)
            print(f"✔ Season {season} gamelogs -> {csv_path}")
            continue  # πάμε επόμενη season

        # --kind season (όπως πριν)
        # εδώ βάλε ό,τι είχες (π.χ. fetch_from_incrowd για season averages)
        df = fetch_from_incrowd(season, competition, mode)

    except Exception as e:
        print(f"[Σφάλμα] Season {season}: {e}", file=sys.stderr)
        continue

    # enrichment + write όπως ήδη κάνεις…
    try:
        df = add_advanced_metrics(df)
    except Exception as _e:
        print(f"[Προειδοποίηση] Advanced metrics: {_e}")
    csv_path, xlsx_path, db_path = write_outputs(df, season, mode, out_dir)
    print(f"✔ Season {season}:\n  - CSV: {csv_path}\n  - XLSX: {xlsx_path}\n  - DB: {db_path}")


        # Advanced metrics enrichment (best-effort)
        try:
            if df is not None and len(df) > 0:
                df = add_advanced_metrics(df)
        except Exception as _e:
            print(f"[Προειδοποίηση] Advanced metrics: {_e}")

        csv_path, xlsx_path, db_path = write_outputs(df, season, mode, out_dir)
        if csv_path:
            print(f"✔ Season {season}:")
            print(f"  - CSV:  {csv_path}")
            print(f"  - XLSX: {xlsx_path}")
            print(f"  - DB:   {db_path}")

    if used_raw:
        print("\n(Χρησιμοποιήθηκε raw HTTP/scrape fallback — αν κάτι δεν κατέβηκε, έλεγξε endpoints/HTML.)")


if __name__ == "__main__":
    main()
