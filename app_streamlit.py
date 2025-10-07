# app_streamlit.py
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import os, re, pandas as pd, streamlit as st
from urllib.parse import urlencode

SEASON = "2025"  # ή E2025 αν έτσι δουλεύεις
avg_path = f"out/players_{SEASON}_perGame.csv"
urls_path = f"out/player_urls_{SEASON}.csv"

df_avg = pd.read_csv(avg_path)
df_urls = pd.read_csv(urls_path)
df = df_avg.merge(df_urls[["player_code","player_url"]], on="player_code", how="left")

qp = st.query_params
player_code = qp.get("player_code")
player_code = st.query_params.get("player_code")

if player_code:
    # αν έχεις ήδη δική σου page_player(player_code), κάλεσέ την και σταμάτα:
    try:
        page_player(player_code)  # <-- η δική σου function αν υπάρχει
        st.stop()
    except NameError:
        # αλλιώς, fallback: διάβασε out/player_urls_2025.csv και δείξε gamelog με read_html
        import pandas as pd, re, requests
        urls = pd.read_csv("out/player_urls_2025.csv")
        row = urls[urls["player_code"].astype(str) == str(player_code)].head(1)
        if row.empty or not str(row.iloc[0].get("player_url","")).strip():
            st.error("Δεν βρέθηκε player_url για αυτόν τον παίκτη."); st.stop()
        player_url = row.iloc[0]["player_url"]
        player_name = row.iloc[0].get("Player", str(player_code))

        st.title(f"{player_name} — Αναλυτικά (Game-by-Game)")
        html = requests.get(player_url, headers={"User-Agent":"eurol-app/1.0"}, timeout=20).text
        tables = pd.read_html(html)
        def score(df):
            cols = [re.sub(r"\\W+","",str(c).lower()) for c in df.columns]
            keys = ["pir","min","λεπ","pts","πον","date","ημ","opponent","αντιπ"]
            return sum(any(k in c for c in cols) for k in keys)
        gl = max(tables, key=score)
        st.dataframe(gl, use_container_width=True)
        for c in ["Πόντοι","PTS","PIR","pir"]:
            if c in gl.columns:
                st.line_chart(gl[c])
        st.stop()

@st.cache_data(ttl=3600)
def scrape_gamelog_table(player_url: str) -> pd.DataFrame:
    import requests, re
    from bs4 import BeautifulSoup
    import pandas as pd

    headers = {
        "User-Agent": "eurol-app/1.1 (+stats)",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "el,en;q=0.8",
    }

    def _read_best_table(html: str) -> pd.DataFrame | None:
        # 1) δοκίμασε κατευθείαν όλα τα tables
        try:
            tables = pd.read_html(html)
        except ValueError:
            tables = []
        # 2) fallback: εντόπισε πρώτα <table> με soup και διάβασε στοχευμένα
        if not tables:
            soup = BeautifulSoup(html, "lxml")
            t_candidates = soup.find_all("table")
            for t in t_candidates:
                try:
                    part = pd.read_html(str(t))[0]
                    tables.append(part)
                except Exception:
                    continue
        if not tables:
            return None
        # επίλεξε το “πιο gamelog” table
        def score(df):
            cols = [re.sub(r"\W+", "", str(c).lower()) for c in df.columns]
            keys = ["pir","min","λεπ","pts","πον","date","ημ","opponent","αντιπ"]
            return sum(any(k in c for c in cols) for k in keys)
        return max(tables, key=score)

    # Δοκίμασε διάφορες παραλλαγές URL
    variants = []
    u = player_url.strip()
    variants.append(u)
    variants.append(u.rstrip("/"))
    if not u.endswith("/"):
        variants.append(u + "/")
    if "/el/" in u:
        variants.append(u.replace("/el/", "/en/"))
    elif "/en/" in u:
        variants.append(u.replace("/en/", "/el/"))

    s = requests.Session()
    for v in variants:
        try:
            r = s.get(v, headers=headers, timeout=20, allow_redirects=True)
            html = r.text
            best = _read_best_table(html)
            if best is not None and not best.empty:
                return best
        except Exception:
            continue

    raise ValueError("No tables found")


def link_for(pcode:str) -> str:
    return "?" + urlencode({"player_code": pcode})
# ---------- ΡΥΘΜΙΣΕΙΣ ----------
OUT_DIR = Path("out")
st.set_page_config(page_title="EuroLeague Fantasy – Player Game Logs", layout="wide")


# ---------- ΒΟΗΘΗΤΙΚΑ ----------
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    for kwargs in ({}, {"sep": ";"}, {"encoding": "utf-16"}):
        try:
            df = pd.read_csv(path, **kwargs)
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception:
            continue
    return None
    
def scale_0_100_robust(s: pd.Series, low_q=0.05, high_q=0.95) -> pd.Series:
    """Robust min–max σε 0..100 χρησιμοποιώντας τα quantiles (κόβει outliers)."""
    s = s.astype(float).replace([np.inf, -np.inf], np.nan)
    lo, hi = s.quantile(low_q), s.quantile(high_q)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(50.0, index=s.index)  # ουδέτερο αν δεν γίνεται κλίμακα
    s_clip = s.clip(lo, hi)
    return ((s_clip - lo) / (hi - lo) * 100.0)
    
def universal_score_raw(df: pd.DataFrame) -> pd.Series:
    avail = (df.get("Min", 0).fillna(0) / 30.0).clip(0.6, 1.2)

    def _minmax(s):
        s = s.replace([np.inf, -np.inf], np.nan)
        if s.max(skipna=True) == s.min(skipna=True):
            return pd.Series(0.5, index=s.index)
        return (s - s.min(skipna=True)) / (s.max(skipna=True) - s.min(skipna=True))

    ts       = df.get("TS%", 0).clip(0, 1)
    usage    = _minmax(df.get("Usage/min", 0))
    trm      = _minmax(df.get("TR/min", 0))
    astm     = _minmax(df.get("AST/min", 0))
    fdm      = _minmax(df.get("FD/min", 0))
    stocksm  = _minmax(df.get("Stocks/min", 0))
    tom_good = 1 - _minmax(df.get("TO/min", 0))

    raw = (
        0.20*usage + 0.18*ts + 0.18*trm + 0.16*astm +
        0.12*fdm   + 0.10*stocksm + 0.06*tom_good
    )
    return raw * avail

def universal_score(df: pd.DataFrame) -> pd.Series:
    # Availability (όσο κοντά στα 30' τόσο καλύτερα)
    avail = (df.get("Min", 0).fillna(0) / 30.0).clip(0.6, 1.2)

    # Normalize components (0..1)
    def _minmax(s):
        s = s.replace([np.inf, -np.inf], np.nan)
        if s.max(skipna=True) == s.min(skipna=True):
            return pd.Series(0.5, index=s.index)
        return (s - s.min(skipna=True)) / (s.max(skipna=True) - s.min(skipna=True))

    ts       = df.get("TS%", 0).clip(0, 1)               # 0..1
    usage    = _minmax(df.get("Usage/min", 0))
    trm      = _minmax(df.get("TR/min", 0))
    astm     = _minmax(df.get("AST/min", 0))
    fdm      = _minmax(df.get("FD/min", 0))
    stocksm  = _minmax(df.get("Stocks/min", 0))
    tom_good = 1 - _minmax(df.get("TO/min", 0))          # λιγότερα λάθη = καλύτερα

    # Ισορροπημένα βάρη για “ποιος θα γράψει PIR”
    raw = (
        0.20*usage + 0.18*ts + 0.18*trm + 0.16*astm +
        0.12*fdm   + 0.10*stocksm + 0.06*tom_good
    )

    score = raw * avail
    return (100 * score / score.max()).round(1) if score.max() > 0 else score

def make_players_path(season: str, mode: str) -> Path:
    return OUT_DIR / f"players_{season}_{mode}.csv"


def make_gamelogs_path(season: str, mode: str) -> Path:
    return OUT_DIR / f"player_gamelogs_{season}_{mode}.csv"


def _first_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------- NORMALIZATION ----------
def normalize_players_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str], list[str]]:
    """Map aliases -> ζητούμενες στήλες & κράτα μόνο αυτές που έχουμε."""

    # --- Rename aliases σε canonical ονόματα που ζητάς ---
    rename_map: Dict[str, str] = {}

    # IDs / meta
    for a in ["player_code", "code", "playerCode"]:
        if a in df.columns: rename_map[a] = "player_code"
    for a in ["player_name", "name", "playerName"]:
        if a in df.columns: rename_map[a] = "Player"
    for a in ["team_code", "player_team_code", "teamCode", "team"]:
        if a in df.columns: rename_map[a] = "Team"
    for a in ["team_name", "player_team_name", "teamName"]:
        if a in df.columns and "Team" not in rename_map.values(): rename_map[a] = "Team"

    # πιθανές στήλες θέσης
    for a in ["position", "player_position", "pos", "PlayerPosition", "Position"]:
        if a in df.columns: rename_map[a] = "Position"

    # Games
    for a in ["gamesPlayed", "GP", "games", "G"]:
        if a in df.columns: rename_map[a] = "GP"
    for a in ["gamesStarted", "GS"]:
        if a in df.columns: rename_map[a] = "GS"

    # Minutes / PIR
    for a in ["minutesPlayed", "MIN", "Minutes"]:
        if a in df.columns: rename_map[a] = "Min"
    for a in ["pir", "PIR", "EFF", "efficiency"]:
        if a in df.columns: rename_map[a] = "PIR"

    # Shooting splits
    # 2P
    for a in ["twoPointersMade", "2PM"]:
        if a in df.columns: rename_map[a] = "2PM"
    for a in ["twoPointersAttempted", "2PA"]:
        if a in df.columns: rename_map[a] = "2PA"
    for a in ["twoPointersPercentage", "2P%"]:
        if a in df.columns: rename_map[a] = "2P%"

    # 3P
    for a in ["threePointersMade", "3PM"]:
        if a in df.columns: rename_map[a] = "3PM"
    for a in ["threePointersAttempted", "3PA"]:
        if a in df.columns: rename_map[a] = "3PA"
    for a in ["threePointersPercentage", "3P%"]:
        if a in df.columns: rename_map[a] = "3P%"

    # FT
    for a in ["freeThrowsMade", "FTM"]:
        if a in df.columns: rename_map[a] = "FTM"
    for a in ["freeThrowsAttempted", "FTA"]:
        if a in df.columns: rename_map[a] = "FTA"
    for a in ["freeThrowsPercentage", "FT%"]:
        if a in df.columns: rename_map[a] = "FT%"

    # Rebounds
    for a in ["offensiveRebounds", "OR", "OREB"]:
        if a in df.columns: rename_map[a] = "OR"
    for a in ["defensiveRebounds", "DR", "DREB"]:
        if a in df.columns: rename_map[a] = "DR"
    for a in ["totalRebounds", "REB", "TR", "TRB"]:
        if a in df.columns: rename_map[a] = "TR"

    # Points / playmaking / defense / fouls
    for a in ["pointsScored", "PTS", "Points"]:
        if a in df.columns: rename_map[a] = "PTS"
    for a in ["assists", "AST"]:
        if a in df.columns: rename_map[a] = "AST"
    for a in ["steals", "STL"]:
        if a in df.columns: rename_map[a] = "STL"
    for a in ["turnovers", "TOV", "TO"]:
        if a in df.columns: rename_map[a] = "TO"
    for a in ["blocks", "BLK"]:
        if a in df.columns: rename_map[a] = "BLK"
    for a in ["blocksAgainst", "BLKA", "BLK_AG"]:
        if a in df.columns: rename_map[a] = "BLKA"
    for a in ["foulsCommited", "foulsCommitted", "FC", "FLS_CM"]:
        if a in df.columns: rename_map[a] = "FC"
    for a in ["foulsDrawn", "FD", "FLS_RV"]:
        if a in df.columns: rename_map[a] = "FD"

    # Season / competition (τα κρατάμε κρυφά για join)
    for a in ["season", "Season"]:
        if a in df.columns: rename_map[a] = "season"
    for a in ["competition", "Competition"]:
        if a in df.columns: rename_map[a] = "competition"

    df = df.rename(columns=rename_map)

    # Γέμισε ελάχιστα πεδία αν λείπουν
    for c in ["Player", "player_code", "Team"]:
        if c not in df.columns: df[c] = None

    # Υπολόγισε TR αν λείπει αλλά έχουμε OR+DR
    if "TR" not in df.columns and all(c in df.columns for c in ["OR", "DR"]):
        df["TR"] = df["OR"].fillna(0) + df["DR"].fillna(0)

    # Ζητούμενη σειρά base στηλών
    target_cols = [
        "Player", "Team",
        "GP", "GS", "Min", "PTS",
        "2PM", "2PA", "2P%",
        "3PM", "3PA", "3P%",
        "FTM", "FTA", "FT%",
        "OR", "DR", "TR",
        "AST", "STL", "TO", "BLK", "BLKA",
        "FC", "FD", "PIR",
        "BCI", "Stability", "Form3",  # advanced που θα συμπληρωθούν
    ]

    # Ταξινόμηση by PIR αν υπάρχει
    if "PIR" in df.columns:
        df = df.sort_values("PIR", ascending=False, na_position="last")

    keep = [c for c in target_cols if c in df.columns and c not in ["BCI", "Stability", "Form3"]]
    return df, keep, target_cols


def normalize_gamelogs_df(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for a in ["player_code", "code", "playerCode"]:
        if a in df.columns: ren[a] = "player_code"
    for a in ["player_name", "name", "playerName"]:
        if a in df.columns: ren[a] = "Player"
    for a in ["team_code", "teamCode", "Team"]:
        if a in df.columns: ren[a] = "Team"
    for a in ["opponent", "opponent_code", "Opp", "Opponent"]:
        if a in df.columns: ren[a] = "opponent"
    for a in ["game_date", "date", "gameDate", "Date"]:
        if a in df.columns: ren[a] = "game_date"
    df = df.rename(columns=ren)

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    def _alias(dst, *alts):
        if dst not in df.columns:
            for a in alts:
                if a in df.columns:
                    df.rename(columns={a: dst}, inplace=True)
                    break

    _alias("MIN", "minutesPlayed", "Min", "Minutes")
    _alias("PTS", "pointsScored", "Points")
    _alias("TR", "totalRebounds", "REB", "TRB")
    _alias("AST", "assists")
    _alias("STL", "steals")
    _alias("TO", "turnovers", "TOV")
    _alias("BLK", "blocks")
    _alias("FC", "foulsCommitted", "foulsCommited", "FLS_CM")
    _alias("FD", "foulsDrawn", "FLS_RV", "FD")
    _alias("PIR", "pir", "EFF", "efficiency")
    return df


# ---------- ADVANCED ----------
def per_min(x: pd.Series, minutes: pd.Series) -> pd.Series:
    x = x.fillna(0)
    m = minutes.replace(0, np.nan)
    return (x / m).fillna(0)


def compute_attempts_from_pct(made: pd.Series, pct: pd.Series) -> pd.Series:
    """Αν λείπουν attempts αλλά έχουμε makes & %, εκτίμηση A = M / (pct/100)."""
    made = made.astype(float).fillna(0)
    p = pct.astype(float).replace(0, np.nan)
    return (made / (p / 100.0)).round(1).fillna(0)


def add_feature_columns(players_df: pd.DataFrame) -> pd.DataFrame:
    df = players_df.copy()

    # Εξασφάλισε attempts (2PA, 3PA, FTA) αν λείπουν
    if "2PA" not in df.columns and all(c in df.columns for c in ["2PM", "2P%"]):
        df["2PA"] = compute_attempts_from_pct(df["2PM"], df["2P%"])
    if "3PA" not in df.columns and all(c in df.columns for c in ["3PM", "3P%"]):
        df["3PA"] = compute_attempts_from_pct(df["3PM"], df["3P%"])
    # FTA συνήθως υπάρχει· αν λείπει αλλά υπάρχει FT%/FTM:
    if "FTA" not in df.columns and all(c in df.columns for c in ["FTM", "FT%"]):
        df["FTA"] = compute_attempts_from_pct(df["FTM"], df["FT%"])

    # Προσπάθειες/FG totals
    df["FGA"] = df.get("2PA", 0).fillna(0) + df.get("3PA", 0).fillna(0)

    # eFG% = (2PM + 1.5*3PM) / FGA
    efg_num = df.get("2PM", 0).fillna(0) + 1.5 * df.get("3PM", 0).fillna(0)
    df["eFG%"] = (efg_num / df["FGA"].replace(0, np.nan)).fillna(0)

    # TS% ~ PTS / (2*(FGA + 0.44*FTA))
    denom = 2 * (df["FGA"] + 0.44 * df.get("FTA", 0).fillna(0))
    df["TS%"] = (df.get("PTS", 0).fillna(0) / denom.replace(0, np.nan)).fillna(0)

    # FT Rate
    df["FTR"] = (df.get("FTA", 0).fillna(0) / df["FGA"].replace(0, np.nan)).fillna(0)

    # Per-minute metrics
    df["PTS/min"] = per_min(df.get("PTS", 0), df.get("Min", 0))
    df["TR/min"]  = per_min(df.get("TR", 0),  df.get("Min", 0))
    df["AST/min"] = per_min(df.get("AST", 0), df.get("Min", 0))
    df["FD/min"]  = per_min(df.get("FD", 0),  df.get("Min", 0))
    df["TO/min"]  = per_min(df.get("TO", 0),  df.get("Min", 0))
    df["STL/min"] = per_min(df.get("STL", 0), df.get("Min", 0))
    df["BLK/min"] = per_min(df.get("BLK", 0), df.get("Min", 0))
    df["Stocks/min"] = df["STL/min"] + df["BLK/min"]

    # Usage/min proxy = (2PA + 3PA + 0.44*FTA + TO) / Min
    usage_numer = df.get("2PA", 0).fillna(0) + df.get("3PA", 0).fillna(0) + 0.44 * df.get("FTA", 0).fillna(0) + df.get("TO", 0).fillna(0)
    df["Usage/min"] = per_min(usage_numer, df.get("Min", 0))

    return df


def compute_bci(players_df: pd.DataFrame) -> pd.Series:
    PTS_pm = players_df["PTS/min"]
    TR_pm  = players_df["TR/min"]
    AST_pm = players_df["AST/min"]
    FD_pm  = players_df["FD/min"]
    PIR_pm = per_min(players_df.get("PIR", 0), players_df.get("Min", 0))

    raw = 0.35*PTS_pm + 0.25*TR_pm + 0.25*AST_pm + 0.10*FD_pm + 0.05*PIR_pm
    min_bonus = (players_df.get("Min", 0).fillna(0) / 30.0).clip(0.6, 1.2)
    raw = raw * min_bonus
    if raw.max() > 0:
        bci = 100 * raw / raw.max()
    else:
        bci = raw
    return bci.round(1)


def compute_stability_form3(players_df: pd.DataFrame, gl_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    adv = pd.DataFrame(index=players_df.index)
    stability_map, form3_map = {}, {}

    if gl_df is not None and not gl_df.empty:
        key = "player_code" if "player_code" in gl_df.columns else ("Player" if "Player" in gl_df.columns else None)
        if key is not None:
            for pid, g in gl_df.groupby(key):
                g = g.sort_values("game_date")
                pir = g.get("PIR")
                if pir is None or pir.dropna().empty:
                    continue
                last6 = pir.dropna().tail(6)
                if len(last6) >= 3 and last6.mean() != 0:
                    cv = last6.std(ddof=0) / abs(last6.mean())
                    stab = (1.0 / (1.0 + cv)) * 100.0
                    stability_map[pid] = float(np.clip(stab, 0, 100))
                last3 = pir.dropna().tail(3)
                if len(last3) > 0:
                    form3_map[pid] = float(last3.mean())

            if "player_code" in players_df.columns and key == "player_code":
                adv["Stability"] = players_df["player_code"].map(stability_map)
                adv["Form3"] = players_df["player_code"].map(form3_map)
            elif "Player" in players_df.columns and key == "Player":
                adv["Stability"] = players_df["Player"].map(stability_map)
                adv["Form3"] = players_df["Player"].map(form3_map)

    adv["Stability"] = adv.get("Stability", pd.Series(index=players_df.index)).round(1)
    adv["Form3"] = adv.get("Form3", pd.Series(index=players_df.index)).round(1)
    return adv


# ---------- POSITION-AWARE RANKING ----------
def minmax(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.max(skipna=True) == s.min(skipna=True):
        return pd.Series(0.5, index=s.index)  # επίπεδο -> ουδέτερο
    return (s - s.min(skipna=True)) / (s.max(skipna=True) - s.min(skipna=True))


def ensure_position_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Position" in df.columns:
        return df
    st.warning(
        "Δεν βρέθηκε στήλη θέσης (Position). Μπορείς να ορίσεις manual mapping "
        "στην ενότητα: **Position mapping (optional)**. Διαφορετικά ο πίνακας προτάσεων δεν θα εμφανιστεί."
    )
    df["Position"] = None
    return df


def apply_manual_position_mapping(df: pd.DataFrame, mapping_text: str) -> pd.DataFrame:
    """
    mapping_text: lines μορφής  PlayerName,POS  ή  player_code,POS
    POS ∈ {G, F, C} (επιτρέπεται και σύνθετο π.χ. 'G/F' — θα πάρουμε το πρώτο γράμμα)
    """
    if not mapping_text.strip():
        return df
    mp = {}
    for line in mapping_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 2:
            key, pos = parts[0], parts[1]
            mp[key] = pos

    # πρώτα με player_code αν υπάρχει
    if "player_code" in df.columns:
        df.loc[df["player_code"].astype(str).isin(mp.keys()), "Position"] = (
            df["player_code"].astype(str).map(mp)
        )
    # μετά με Player name
    if "Player" in df.columns:
        df.loc[df["Player"].astype(str).isin(mp.keys()), "Position"] = (
            df["Player"].astype(str).map(mp)
        )

    # καθάρισε σε G/F/C (αν δόθηκε G/F, πάρε το πρώτο)
    df["Position"] = df["Position"].apply(lambda x: str(x)[0].upper() if pd.notna(x) else x)
    return df


def position_score(df: pd.DataFrame, role: str) -> pd.Series:
    """
    Φτιάχνει σύνθετο score 0..100 ανά ρόλο (G/F/C) από normalized components.
    Χρησιμοποιεί season-based features (per-minute + efficiency).
    """

    # Availability multiplier: πιο κοντά στα 30' → καλύτερα
    avail = (df.get("Min", 0).fillna(0) / 30.0).clip(0.6, 1.2)

    # Θετικά components (normalize 0..1)
    ts = df.get("TS%", 0).clip(0, 1)              # 0..1
    ef = df.get("eFG%", 0).clip(0, 1)             # 0..1
    twop = (df.get("2P%", 0)/100.0).clip(0, 1)    # 0..1
    ftr = df.get("FTR", 0).clip(0, None)          # >0
    usage = minmax(df.get("Usage/min", 0))
    trm = minmax(df.get("TR/min", 0))
    astm = minmax(df.get("AST/min", 0))
    fdm = minmax(df.get("FD/min", 0))
    stocksm = minmax(df.get("Stocks/min", 0))

    # Αρνητικό: fewer is better → 1 - norm
    tom = 1 - minmax(df.get("TO/min", 0))

    if role == "G":
        # Guards: usage & creation & efficiency, draw fouls, protect ball, some stocks
        raw = (
            0.25*usage + 0.25*astm + 0.20*ts + 0.10*fdm +
            0.10*ef + 0.05*stocksm + 0.05*tom
        )
    elif role == "F":
        # Forwards: glass + usage + efficiency + creation + fouls + stocks, penalty TO
        raw = (
            0.25*trm + 0.20*usage + 0.15*ts + 0.10*ef +
            0.10*fdm + 0.10*astm + 0.05*stocksm + 0.05*tom
        )
    else:  # "C"
        # Centers: rebounds + rim protection + 2P% + fouls drawn, then usage/TS, penalty TO
        raw = (
            0.30*trm + 0.20*stocksm + 0.15*twop + 0.15*fdm +
            0.10*usage + 0.05*ts + 0.05*tom
        )

    score = raw * avail
    return (100 * score / score.max()) if score.max() > 0 else (score * 0)


def build_position_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {}
    for role, topk in [("C", 10), ("F", 15), ("G", 20)]:
        part = df[df["Position"].fillna("").str.upper().str.startswith(role)].copy()
        if part.empty:
            out[role] = part
            continue
        part[f"{role}_Score"] = position_score(part, role).round(1)
        part = part.sort_values(f"{role}_Score", ascending=False, na_position="last")
        keep_cols = [
            "Player", "Team", "Position", "Min", "PTS", "TR", "AST", "STL", "BLK", "FD", "TO",
            "TS%", "eFG%", "FTR", "Usage/min", "TR/min", "AST/min", "FD/min", "Stocks/min", "TO/min",
            "PIR", "BCI", f"{role}_Score"
        ]
        keep_cols = [c for c in keep_cols if c in part.columns]
        out[role] = part[keep_cols].head(topk).reset_index(drop=True)
    return out


# ---------- UI ----------
st.title("EuroLeague Fantasy – Player Game Logs")

# Επιλογές header
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    season = st.selectbox("Season", ["2025"], index=0)
with c2:
    competition = st.selectbox("Competition", ["E", "U"], index=0)
with c3:
    mode = st.selectbox("Mode", ["perGame"], index=0)

players_path = make_players_path(season, mode)
gamelogs_path = make_gamelogs_path(season, mode)

players_df_raw = load_csv(players_path)
gamelogs_df_raw = load_csv(gamelogs_path)

st.caption(
    f"📄 Averages: `{players_path}` — "
    f"{'OK' if players_df_raw is not None else 'ΔΕΝ ΒΡΕΘΗΚΕ'}  |  "
    f"📄 Gamelogs: `{gamelogs_path}` — "
    f"{'OK' if gamelogs_df_raw is not None else 'ΔΕΝ ΒΡΕΘΗΚΕ'}"
)

if players_df_raw is None:
    st.error(f"Δεν βρέθηκε το αρχείο season averages: `{players_path}`.")
    st.stop()

# Normalize + base columns
players_df_norm, base_cols, target_cols = normalize_players_df(players_df_raw)

# Gamelogs (optional)
gamelogs_df = None
if gamelogs_df_raw is not None and not gamelogs_df_raw.empty:
    gamelogs_df = normalize_gamelogs_df(gamelogs_df_raw.copy())

# Add features + BCI + Stability/Form3
players_df_feat = add_feature_columns(players_df_norm)
players_df_feat["BCI"] = compute_bci(players_df_feat)

adv_sf = compute_stability_form3(players_df_feat, gamelogs_df)
players_df = players_df_feat.join(adv_sf)

players_df["All_Score_raw"] = universal_score_raw(players_df)
max_all = players_df["All_Score_raw"].max()
players_df["All_Score"] = (players_df["All_Score_raw"] / max_all * 100).round(1)

# --- ΝΕΟ: All_Score ως global advanced στήλη ---
players_df["All_Score"] = universal_score(players_df)
# --- PredictScore (0..100): blend All_Score + Form3 + Stability, με gates ---
form3_norm = scale_0_100_robust(players_df.get("Form3", pd.Series(index=players_df.index)))
stab_norm  = players_df.get("Stability", pd.Series(index=players_df.index)).astype(float)
players_df["PredictScore"] = (
    0.60*players_df["All_Score"].fillna(0) +
    0.30*form3_norm.fillna(50) +
    0.10*stab_norm.fillna(50)
) * (players_df.get("Min", 0).fillna(0)/30.0).clip(0.6, 1.15)
players_df["PredictScore"] = players_df["PredictScore"].clip(0, 100).round(1)

# Βάρη (μπορείς να τα αλλάξεις εύκολα)
w_all, w_form3, w_stab = 0.60, 0.30, 0.10

base = (
    w_all   * players_df["All_Score"].astype(float).fillna(0) +
    w_form3 * form3_norm.fillna(50.0) +   # αν λείπουν gamelogs, ουδέτερο ~50
    w_stab  * stab_norm.fillna(50.0)
)

# Gates: λεπτά & μικρό penalty για χαμηλό GP
min_mult = (players_df.get("Min", 0).fillna(0) / 30.0).clip(0.6, 1.15)
gp_mult  = np.where(players_df.get("GP", 0).fillna(0) >= 3, 1.0, 0.90)

players_df["PredictScore"] = (base * min_mult * gp_mult).clip(0, 100).round(1)


# Εξασφάλισε να υπάρχουν πάντα στήλες Stability/Form3
for _col in ["Stability", "Form3"]:
    if _col not in players_df.columns:
        players_df[_col] = np.nan

# Φίλτρα / αναζήτηση
teams_list = ["(Όλες)"] + sorted(players_df.get("Team", pd.Series(dtype=str)).dropna().astype(str).unique(), key=lambda x: x.lower())
f1, f2, f3 = st.columns([2, 1, 1])
with f1:
    q = st.text_input("🔎 Live search (όνομα/κωδικός/ομάδα)", "")
with f2:
    team_sel = st.selectbox("Ομάδα", teams_list, index=0)
with f3:
    min_gp = st.number_input("Min GP", min_value=0, max_value=50, value=0, step=1)

def filter_players(df: pd.DataFrame, q: str, team: str, min_gp: int) -> pd.DataFrame:
    res = df.copy()
    if q:
        qlow = q.lower().strip()
        res = res[
            res.get("Player", pd.Series(index=res.index, dtype=str)).fillna("").str.lower().str.contains(qlow)
            | res.get("player_code", pd.Series(index=res.index, dtype=str)).fillna("").astype(str).str.contains(qlow)
            | res.get("Team", pd.Series(index=res.index, dtype=str)).fillna("").str.lower().str.contains(qlow)
        ]
    if team and team != "(Όλες)":
        res = res[res.get("Team", pd.Series(index=res.index, dtype=str)).fillna("").str.lower().eq(team.lower())]
    if "GP" in res.columns and min_gp > 0:
        res = res[res["GP"].fillna(0) >= min_gp]
    return res

filtered_players = filter_players(players_df, q, team_sel, min_gp)

final_cols = []
for c in target_cols:
    if c in filtered_players.columns:
        final_cols.append(c)
# πρόσθεσε All_Score αν δεν υπάρχει ήδη
# Season table – πρόσθεσε Always τις extra στήλες αν υπάρχουν
if "All_Score" in filtered_players.columns and "All_Score" not in final_cols:
    final_cols.append("All_Score")
if "PredictScore" in filtered_players.columns and "PredictScore" not in final_cols:
    final_cols.append("PredictScore")

st.subheader("Season Averages (με τις ζητούμενες στήλες + Advanced)")

# ========= ΜΟΝΗ ΑΛΛΑΓΗ: κάνουμε τη στήλη Player clickable & render ως HTML =========
# --- μικρό font για τον πίνακα ---
st.markdown("""
<style>
.small-table table { font-size: 14px; }
.small-table th, .small-table td { padding: 6px 10px; }
</style>
""", unsafe_allow_html=True)


def _plink_player(code, name):
    from urllib.parse import urlencode
    qs = urlencode({"player_code": str(code)})
    return f'<a href="?{qs}" style="text-decoration:none;">{name}</a>'

# 1) ξεκινάμε από το filtered_players που ήδη έχεις
table_df = filtered_players.copy()

# 2) κάνε τη στήλη Player clickable (θέλει και player_code)
if "Player" in table_df.columns and "player_code" in table_df.columns:
    table_df["Player"] = [
        _plink_player(c, n) for c, n in zip(table_df["player_code"], table_df["Player"])
    ]

# 3) ποια columns θα δείξουμε (με βάση final_cols)
display_cols = [c for c in final_cols if c in table_df.columns]

# 4) Show more / Show less
if "show_all" not in st.session_state:
    st.session_state["show_all"] = False

if st.session_state["show_all"]:
    display_df = table_df[display_cols].reset_index(drop=True)
    if st.button("Show less", key="less"):
        st.session_state["show_all"] = False
        st.rerun()
else:
    display_df = table_df[display_cols].head(30).reset_index(drop=True)
    if st.button("Show more", key="more"):
        st.session_state["show_all"] = True
        st.rerun()

# 5) render με μικρότερο font
st.markdown(
    f"<div class='small-table'>{display_df.to_html(index=False, escape=False)}</div>",
    unsafe_allow_html=True,
)


# ========= ΤΕΛΟΣ ΑΛΛΑΓΗΣ =========




# ----------------- ANALYTICS TABS -----------------
tabs = st.tabs([
    "📈 Player details (gamelogs)",
    "🧮 Advanced features",
    "🏆 Προτεινόμενα Picks (G/F/C)",
    "🔥 Top 30 (All)"
])


# --- TAB 1: Player details ---
with tabs[0]:
    left, right = st.columns([1, 2])
    with left:
        st.markdown("### Επιλογή παίκτη")
        options = (
            filtered_players[["Player", "player_code"]]
            .dropna(subset=["Player"])
            .drop_duplicates()
            .sort_values("Player")
            .assign(label=lambda d: d["Player"] + "  (" + d["player_code"].astype(str) + ")")
        )
        if len(options) == 0:
            selected_label = None
            st.info("Κανένα αποτέλεσμα με τα τρέχοντα φίλτρα.")
        else:
            selected_label = st.selectbox("Διάλεξε παίκτη", options["label"].tolist(), index=0, key="player_select")

            selected_player_code = None
            if selected_label:
                sel_row = options[options["label"] == selected_label].iloc[0]
                selected_player_code = str(sel_row["player_code"])

    with right:
        st.markdown("### Αναλυτικά (Game-by-Game)")
        if gamelogs_df is None or gamelogs_df.empty:
            st.warning("Δεν υπάρχουν gamelogs στο repository για να εμφανιστούν αναλυτικά.")
        else:
            player_gl = pd.DataFrame()
            if selected_label:
                if "player_code" in gamelogs_df.columns and selected_player_code is not None:
                    player_gl = gamelogs_df[gamelogs_df["player_code"].astype(str) == selected_player_code]
                if player_gl.empty and "Player" in gamelogs_df.columns:
                    p_name = sel_row["Player"]
                    player_gl = gamelogs_df[gamelogs_df["Player"].astype(str).str.lower() == str(p_name).lower()]

            if selected_label and not player_gl.empty:
                csum1, csum2, csum3 = st.columns(3)
                with csum1: st.metric("Games", len(player_gl))
                with csum2: st.metric("PTS (avg)", round(player_gl.get("PTS", pd.Series([0])).mean(), 2))
                with csum3: st.metric("PIR (avg)", round(player_gl.get("PIR", pd.Series([0])).mean(), 2))

                if "game_date" in player_gl.columns:
                    if "PTS" in player_gl.columns:
                        sub = player_gl[["game_date", "PTS"]].dropna().sort_values("game_date").set_index("game_date")
                        st.line_chart(sub, height=180, use_container_width=True)
                    if "PIR" in player_gl.columns:
                        sub = player_gl[["game_date", "PIR"]].dropna().sort_values("game_date").set_index("game_date")
                        st.line_chart(sub, height=180, use_container_width=True)

                gl_cols_pref = ["game_date", "Team", "opponent", "MIN", "PTS", "TR", "AST", "STL", "TO", "BLK", "FC", "FD", "PIR"]
                gl_cols = [c for c in gl_cols_pref if c in player_gl.columns]
                st.dataframe(player_gl[gl_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
            else:
                if selected_label:
                    st.info("Δεν βρέθηκαν gamelogs για τον συγκεκριμένο παίκτη (ή το αρχείο είναι κενό).")

# --- TAB 2: Advanced features table ---
with tabs[1]:
    st.markdown("### Advanced feature set (season-based)")
    feat_cols = [
    "Player", "Team", "Position", "Min", "PIR", "BCI",
    "TS%", "eFG%", "FTR",
    "Usage/min", "PTS/min", "TR/min", "AST/min", "FD/min", "Stocks/min", "TO/min",
    "Stability", "Form3", "All_Score", "PredictScore"
    ]
    feat_cols = [c for c in feat_cols if c in filtered_players.columns]
    #st.dataframe(filtered_players[feat_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

# --- TAB 3: Position-aware Picks ---
with tabs[2]:
    st.markdown("### Position mapping (optional)")
    st.caption("Αν δεν υπάρχει στήλη θέσης στο CSV, μπορείς να δώσεις manual mapping σε μορφή `Player,POS` ή `player_code,POS` (POS ∈ G/F/C)."
               " Μία γραμμή ανά παίκτη. Αν δοθεί π.χ. `G/F`, λαμβάνεται το πρώτο γράμμα.")
    mapping_text = st.text_area("Μετατροπή", value="", height=120)

    players_pos = ensure_position_column(filtered_players.copy())
    if mapping_text.strip():
        players_pos = apply_manual_position_mapping(players_pos, mapping_text)

    # Αν υπάρχουν θέσεις, φτιάξε προτάσεις
    if players_pos["Position"].notna().any():
        pos_tables = build_position_tables(players_pos)

        ctab, ftab, gtab = st.tabs(["🏀 Centers (Top 10)", "🛡️ Forwards (Top 15)", "⚡ Guards (Top 20)"])

        with ctab:
            if pos_tables["C"].empty:
                st.info("Δεν υπάρχουν διαθέσιμοι Centers (λείπει Position mapping; δώσε στο πεδίο από πάνω).")
            else:
                st.dataframe(pos_tables["C"], use_container_width=True, hide_index=True)

        with ftab:
            if pos_tables["F"].empty:
                st.info("Δεν υπάρχουν διαθέσιμοι Forwards (λείπει Position mapping; δώσε στο πεδίο από πάνω).")
            else:
                st.dataframe(pos_tables["F"], use_container_width=True, hide_index=True)

        with gtab:
            if pos_tables["G"].empty:
                st.info("Δεν υπάρχουν διαθέσιμοι Guards (λείπει Position mapping; δώσε στο πεδίο από πάνω).")
            else:
                st.dataframe(pos_tables["G"], use_container_width=True, hide_index=True)
    else:
        st.warning("Δεν υπάρχουν διαθέσιμες θέσεις (Position). Πρόσθεσε mapping για να δεις προτάσεις G/F/C.")
        
# --- TAB 4: Top 30 ανεξάρτητα θέσης ---
with tabs[3]:
    st.markdown("### 🔥 Top 30 (All positions)")
    metric = st.radio("Ταξινόμηση κατά:", ["PredictScore", "All_Score", "PIR"], index=0, horizontal=True)
    sort_col = metric
    top_all = filtered_players.sort_values(sort_col, ascending=False, na_position="last").head(30)

    show_cols = [
        "Player", "Team", "Min", "PIR",
        "TS%", "eFG%", "FTR",
        "Usage/min", "PTS/min", "TR/min", "AST/min", "FD/min", "Stocks/min", "TO/min",
        "BCI", "Stability", "Form3", "All_Score", "PredictScore"
    ]
    show_cols = [c for c in show_cols if c in top_all.columns]
    st.dataframe(top_all[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
