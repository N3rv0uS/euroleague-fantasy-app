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


# ---------------- Helpers & Advanced Metrics ----------------
def _get_col(df, *candidates, default=0):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(default, index=df.index, dtype="float64")


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

    # (B) Fallback: scrape από expanded page
    web_url = (
        f"{base_web}?size=1000&viewType=traditional"
        f"&seasonMode=Range&statisticMode={mode}"
        f"&fromSeasonCode={season_code}&toSeasonCode={season_code}"
        f"&sortDirection=ascending&statistic="
    )
    try:
        print(f"🔎 Fallback to HTML table: {web_url}")
        # Διαβάζουμε με 2 parsers για σιγουριά (lxml, html5lib)
        try:
            tables = pd.read_html(web_url, flavor="lxml")
        except Exception:
            tables = pd.read_html(web_url, flavor="html5lib")

        if not tables:
            print("⚠️ No tables found on expanded page")
            return pd.DataFrame()

        df = tables[0].copy()
        print(f"✅ Scraped HTML table with shape: {df.shape}")

        # Καθαρισμός κεφαλίδων
        df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
        if df.columns[0].strip().lower() in {"#", "unnamed: 0"}:
            df = df.iloc[:, 1:]

        # Map Ελληνικά -> Αγγλικά (όπου υπάρχουν)
        colmap_el_to_en = {
            "Παίκτης": "Player",
            "Ομάδα": "Team",
            "Αγώνες": "GP",
            "Λεπτά": "MIN",
            "Πόντοι": "PTS",
            "Επιθετικά Ριμπάουντ": "OREB",
            "Αμυντικά Ριμπάουντ": "DREB",
            "Συνολικά Ριμπάουντ": "REB",
            "Ασίστ": "AST",
            "Κλεψίματα": "STL",
            "Κοψίματα": "BLK",
            "Λάθη": "TOV",
            "Φάουλ": "PF",
        }
        df.rename(columns={k: v for k, v in colmap_el_to_en.items() if k in df.columns}, inplace=True)

        # Μετατροπή "Λεπτά" mm:ss -> MIN (
