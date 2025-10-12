# app_streamlit.py
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import os, re, pandas as pd, streamlit as st
from urllib.parse import urlencode
import requests
import json, os
import streamlit as st
import requests
import time
from datetime import datetime

st.write("Has GH_PAT:", "GH_PAT" in st.secrets)
SEASON = "2025"  # ή E2025 αν έτσι δουλεύεις
avg_path = f"out/players_{SEASON}_perGame.csv"
urls_path = f"out/player_urls_{SEASON}.csv"

df_avg = pd.read_csv(avg_path)
df_urls = pd.read_csv(urls_path)
df = df_avg.merge(df_urls[["player_code", "player_url"]], on="player_code", how="left")

qp = st.query_params
player_code = qp.get("player_code")
player_code = st.query_params.get("player_code")

def get_file_sha(owner: str, repo: str, path: str, token: str = "") -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}"} if token else {}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()["sha"]

@st.cache_data
def load_csv_by_sha(owner: str, repo: str, path: str, ref_sha: str) -> pd.DataFrame:
    """Cache key = ref_sha => κάθε νέο commit στο αρχείο φρεσκάρει αυτόματα τα data."""
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{path}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))
    
def gh_list_workflows(owner: str, repo: str, token: str):
    """Επιστρέφει λίστα διαθέσιμων workflows στο repo."""
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        st.error(f"GitHub API error {r.status_code}")
        return []
    data = r.json()
    workflows = [
        {"name": wf.get("name"), "path": wf.get("path"), "id": wf.get("id")}
        for wf in data.get("workflows", [])
    ]
    return workflows

def show_player_detail_from_csv(player_code=None, player_name=None):
    # χρησιμοποιεί ήδη φορτωμένα pergame/glogs ή φόρτωσέ τα όπως τα φορτώνεις αλλού
    owner = "N3rv0uS"; repo = "euroleague-fantasy-app"; token = st.secrets.get("GH_PAT","")
    gl_sha = get_file_sha(owner, repo, "out/player_gamelogs_2025_perGame.csv", token)
    gl = load_csv_by_sha(owner, repo, "out/player_gamelogs_2025_perGame.csv", gl_sha)

    # normalizer για key
    def _key(s):
        return (s.astype(str).str.strip().str.upper().str.replace(r"\s+", "", regex=True))

    code_col_gl = "player_code" if "player_code" in gl.columns else "code"

    # βρες τον κωδικό αν ήρθε μόνο όνομα
    if player_code is None and player_name is not None:
        pg_sha = get_file_sha(owner, repo, "out/players_2025_perGame.csv", token)
        per = load_csv_by_sha(owner, repo, "out/players_2025_perGame.csv", pg_sha)
        name_col = "player_name" if "player_name" in per.columns else "Player"
        mask = per[name_col].astype(str).str.contains(str(player_name), case=False, na=False)
        if mask.any():
            player_code = per.loc[mask, "player_code" if "player_code" in per.columns else "code"].iloc[0]

    if player_code is None:
        st.error("Δεν βρέθηκε player_code για εμφάνιση gamelogs.")
        return

    key = _key(pd.Series([player_code])).iloc[0]
    gl["_key"] = _key(gl[code_col_gl])
    rows = gl.loc[gl["_key"] == key].copy()

    # Ταξινόμηση & εμφάνιση
    for cand in ["game_date", "date", "gamedate"]:
        if cand in rows.columns:
            rows[cand] = pd.to_datetime(rows[cand], errors="coerce")
            rows = rows.sort_values(cand)
            break

    cols = [c for c in ["game_date","date","gamedate","opponent","home_away","PIR","PTS","TR","AST","STL","BLK","TO","Min"] if c in rows.columns]
    st.subheader(f"📓 Gamelogs — {player_name or player_code}")
    st.dataframe(rows[cols] if cols else rows, use_container_width=True)

def get_latest_run_status(owner, repo, wf_id, token):
    """Επιστρέφει το status και το conclusion του πιο πρόσφατου run"""
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{wf_id}/runs?per_page=1"
    headers = {"Authorization": f"token {token}"}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return None, f"GitHub API error: {r.status_code}"
    runs = r.json().get("workflow_runs", [])
    if not runs:
        return None, "No runs found"
    run = runs[0]
    return run["status"], run.get("conclusion")

def trigger_actions_dispatch(owner: str, repo: str, workflow_filename: str, token: str, ref: str = "main") -> tuple[bool, str]:
    """
    Καλεί το GitHub Actions workflow_dispatch για να κάνει refresh.
    - owner/repo: π.χ. "myuser", "euroleague-fantasy-app"
    - workflow_filename: π.χ. "euroleague_refresh.yml"
    - token: PAT με scope repo (βάζεις στο st.secrets['GH_PAT'])
    - ref: κλάδος (main ή ό,τι χρησιμοποιείς)
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_filename}/dispatches"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {"ref": ref}
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    ok = (200 <= r.status_code < 300)
    return ok, ("" if ok else f"{r.status_code}: {r.text}")

# --- Expander για manual GitHub refresh ---
with st.expander("🔄 Run GitHub workflow now"):
    owner = "N3rv0uS"
    repo = "euroleague-fantasy-app"
    token = st.secrets.get("GH_PAT", "")

    if token:
        workflows = gh_list_workflows(owner, repo, token)
        choices = {f"{w['name']}  ·  {w['path']}": w["id"] for w in workflows}
        sel = st.selectbox("Διάλεξε workflow", list(choices.keys()), index=len(choices)-1)

        if st.button("▶ Run selected"):
            wf_id = choices[sel]
            ok, err = trigger_actions_dispatch(owner, repo, wf_id, token, ref="main")

            if ok:
                st.success("✅ Dispatch OK – ξεκίνησε το update στο GitHub!")
                progress = st.progress(0)
                status_text = st.empty()
                pct = 0

                # Poll για έως 5 λεπτά (30 * 10s)
                for i in range(30):
                    time.sleep(10)
                    status, conclusion = get_latest_run_status(owner, repo, wf_id, token)

                    if status == "in_progress":
                        pct = min(pct + 5, 90)
                        progress.progress(pct)
                        status_text.info("🔄 Update σε εξέλιξη… (GitHub Actions running)")
                    elif status == "completed":
                        progress.progress(100)
                        if conclusion == "success":
                            status_text.success(f"✅ Update ολοκληρώθηκε επιτυχώς ({datetime.utcnow():%H:%M UTC})")

                            owner = "N3rv0uS"
                            repo  = "euroleague-fantasy-app"
                            token = st.secrets.get("GH_PAT", "")
                            path  = "out/players_2025_perGame.csv"

                            # παλιό SHA (αν υπάρχει)
                            old_sha = st.session_state.get("players_csv_sha", None)

                            # περίμενε έως ~60s να αλλάξει το SHA στο GitHub
                            new_sha = old_sha
                            for _ in range(12):
                                time.sleep(5)
                                try:
                                    new_sha = get_file_sha(owner, repo, path, token)
                                    if new_sha != old_sha:
                                        break
                                except Exception:
                                    pass

                            # καθάρισε cache και ξανατρέξε το app
                            st.cache_data.clear()
                            if new_sha:
                                st.session_state["players_csv_sha"] = new_sha
                            st.rerun()
                        else:
                            status_text.error(f"❌ Απέτυχε ({conclusion})")
                        break
                    else:
                        status_text.write(f"⌛ Περιμένω να ξεκινήσει run... ({status})")
                else:
                    status_text.warning("⚠️ Δεν ήρθε αποτέλεσμα μέσα στο χρονικό όριο (5 λεπτά).")

            else:
                st.error(f"❌ Αποτυχία trigger: {err}")

    else:
        st.error("Λείπει GH_PAT στο Streamlit secrets.")
    # ΚΑΛΕΣΕ ΤΟΝ ΠΙΝΑΚΑ ΕΔΩ
        st.divider()
        render_players_pergame_table()

if player_code:
    try:
        page_player(player_code)
        st.stop()
    except NameError:
        import pandas as pd, re, requests
        urls = pd.read_csv("out/player_urls_2025.csv")
        row = urls[urls["player_code"].astype(str) == str(player_code)].head(1)
        if row.empty or not str(row.iloc[0].get("player_url", "")).strip():
            st.error("Δεν βρέθηκε player_url για αυτόν τον παίκτη.")
            st.stop()
        player_url = row.iloc[0]["player_url"]
        player_name = row.iloc[0].get("Player", str(player_code))

        st.title(f"{player_name} — Αναλυτικά (Game-by-Game)")

        gl = pd.DataFrame()
        # 1) δοκίμασε scraping ΜΟΝΟ αν έχουμε έγκυρο URL
        if player_url and isinstance(player_url, str) and player_url.startswith("http"):
            try:
                gl = scrape_gamelog_table(player_url)
            except Exception as e:
                st.warning(f"Scrape failed: {e}")
        if gl is None or gl.empty:
            st.info("Χρησιμοποιώ gamelogs από το CSV (fallback).")
            # προσαρμοσε 'row' / μεταβλητές όπως τις έχεις
            show_player_detail_from_csv(
                player_code=row.get("player_code") if 'row' in locals() else None,
                player_name=row.get("player_name") if 'row' in locals() else None
            )
        else:
            st.dataframe(gl, use_container_width=True)
            if player_url:
                st.link_button("🔗 Official page", player_url)

        st.dataframe(gl, use_container_width=True)
        for c in ["Πόντοι", "PTS", "PIR", "pir"]:
            if c in gl.columns:
                st.line_chart(gl[c])
        st.stop()
@st.cache_data(ttl=3600)
def scrape_gamelog_table(player_url: str):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd, re

    if not player_url:
        return None

    headers = {
        "User-Agent": "eurol-app/1.2 (+stats)",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "el,en;q=0.8",
    }

    # φτιάξε παραλλαγές URL
    variants = []
    u = player_url.strip()
    variants.append(u)
    if not u.endswith("/"):
        variants.append(u + "/")
    variants.append(u.rstrip("/"))
    if "/el/" in u:
        variants.append(u.replace("/el/", "/en/"))
    if "/en/" in u:
        variants.append(u.replace("/en/", "/el/"))

    s = requests.Session()
    for v in variants:
        try:
            r = s.get(v, headers=headers, timeout=20, allow_redirects=True)
            soup = BeautifulSoup(r.text, "lxml")
            tables = []
            for t in soup.find_all("table"):
                try:
                    part = pd.read_html(str(t))[0]
                    tables.append(part)
                except Exception:
                    continue
            if not tables:
                continue

            def score(df):
                cols = [re.sub(r"\W+", "", str(c).lower()) for c in df.columns]
                keys = ["pir", "min", "λεπ", "pts", "πον", "date", "ημ", "opponent", "αντιπ"]
                return sum(any(k in c for c in cols) for k in keys)

            return max(tables, key=score)
        except Exception:
            continue

    return None   # σωστά μέσα στη function

# ===== Section renderer για τον πίνακα =====
def render_players_pergame_table():
    owner = "N3rv0uS"
    repo  = "euroleague-fantasy-app"
    token = st.secrets.get("GH_PAT", "")

    pergame_path = "out/players_2025_perGame.csv"
    urls_path    = "out/player_urls_2025.csv"

    # --- load per-game (με SHA για σωστό cache bust) ---
    sha_pg = get_file_sha(owner, repo, pergame_path, token)
    st.session_state["players_csv_sha"] = sha_pg
    pergame = load_csv_by_sha(owner, repo, pergame_path, sha_pg)

    # --- load urls csv ---
    sha_urls = get_file_sha(owner, repo, urls_path, token)
    urls_df  = load_csv_by_sha(owner, repo, urls_path, sha_urls)

    # --- merge με ασφαλές key (trim/upper/χωρίς κενά) ---
    def _norm_key(s):
        return (
            s.astype(str)
             .str.strip()
             .str.upper()
             .str.replace(r"\s+", "", regex=True)
        )

    # εντοπισμός σωστού code column στα δύο CSV
    code_per = "player_code" if "player_code" in pergame.columns else "code"
    code_url = "player_code" if "player_code" in urls_df.columns else "code"

    pergame["_key"] = _norm_key(pergame[code_per])
    urls_df["_key"] = _norm_key(urls_df[code_url])

    urls_small = urls_df[["_key", "player_url"]].drop_duplicates("_key")
    pergame = pergame.merge(urls_small, on="_key", how="left").drop(columns=["_key"])

    # --- εμφάνιση ---
    st.subheader("Players — per game (2025)")
    st.caption(f"Source: {owner}/{repo} · `{pergame_path}` sha:{sha_pg[:7]} · urls sha:{sha_urls[:7]}")
    st.dataframe(
        pergame,
        use_container_width=True,
        column_config={
            "player_url": st.column_config.LinkColumn("Player page")
        },
    )

    # (προαιρετικό μικρό debug για BONGA)
    # m = pergame["player_name"].astype(str).str.contains("BONGA", case=False, na=False) if "player_name" in pergame.columns else pergame[code_per].astype(str).str.contains("BONGA", case=False, na=False)
    # st.write(pergame.loc[m, [code_per, "player_name" if "player_name" in pergame.columns else code_per, "player_url"]].head(1))


def show_player_page(player_code: str):
    """Δείξε σελίδα παίκτη: gamelogs από CSV ή scraping."""
    import pandas as pd

    pname = str(player_code)

    try:
        urls_df = pd.read_csv("out/player_urls_2025.csv")
        row = urls_df[urls_df["player_code"].astype(str) == str(player_code)].head(1)
    except Exception:
        row = pd.DataFrame()

    player_url = None
    if not row.empty:
        player_url = str(row.iloc[0].get("player_url", "")).strip()
        pname = row.iloc[0].get("Player", pname)

    st.title(f"{pname} — Αναλυτικά (Game-by-Game)")

    if not player_url:
        st.warning("Δεν βρέθηκε player_url για αυτόν τον παίκτη.")
        return

    gl = scrape_gamelog_table(player_url)

    if gl is None or gl.empty:
        st.warning("Δεν βρέθηκε HTML πίνακας με gamelogs στη σελίδα.")
        st.markdown(f"[Άνοιγμα επίσημου προφίλ]({player_url})")
        return

    st.dataframe(gl, use_container_width=True)
    for c in ["Πόντοι", "PTS", "PIR", "pir"]:
        if c in gl.columns:
            st.line_chart(gl[c])


# ---- Router: αν υπάρχει ?player_code, δείξε τον παίκτη και σταμάτα ----
pc = st.query_params.get("player_code")
if pc:
    show_player_page(pc)
    st.stop()
def link_for(pcode: str) -> str:
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
    rename_map: Dict[str, str] = {}

    # IDs / meta
    for a in ["player_code", "code", "playerCode"]:
        if a in df.columns:
            rename_map[a] = "player_code"
    for a in ["player_name", "name", "playerName"]:
        if a in df.columns:
            rename_map[a] = "Player"
    for a in ["team_code", "player_team_code", "teamCode", "team"]:
        if a in df.columns:
            rename_map[a] = "Team"
    for a in ["team_name", "player_team_name", "teamName"]:
        if a in df.columns and "Team" not in rename_map.values():
            rename_map[a] = "Team"

    # πιθανές στήλες θέσης
    for a in ["position", "player_position", "pos", "PlayerPosition", "Position"]:
        if a in df.columns:
            rename_map[a] = "Position"

    # Games
    for a in ["gamesPlayed", "GP", "games", "G"]:
        if a in df.columns:
            rename_map[a] = "GP"
    for a in ["gamesStarted", "GS"]:
        if a in df.columns:
            rename_map[a] = "GS"

    # Minutes / PIR
    for a in ["minutesPlayed", "MIN", "Minutes"]:
        if a in df.columns:
            rename_map[a] = "Min"
    for a in ["pir", "PIR", "EFF", "efficiency"]:
        if a in df.columns:
            rename_map[a] = "PIR"

    # Shooting splits
    for a in ["twoPointersMade", "2PM"]:
        if a in df.columns:
            rename_map[a] = "2PM"
    for a in ["twoPointersAttempted", "2PA"]:
        if a in df.columns:
            rename_map[a] = "2PA"
    for a in ["twoPointersPercentage", "2P%"]:
        if a in df.columns:
            rename_map[a] = "2P%"

    for a in ["threePointersMade", "3PM"]:
        if a in df.columns:
            rename_map[a] = "3PM"
    for a in ["threePointersAttempted", "3PA"]:
        if a in df.columns:
            rename_map[a] = "3PA"
    for a in ["threePointersPercentage", "3P%"]:
        if a in df.columns:
            rename_map[a] = "3P%"

    for a in ["freeThrowsMade", "FTM"]:
        if a in df.columns:
            rename_map[a] = "FTM"
    for a in ["freeThrowsAttempted", "FTA"]:
        if a in df.columns:
            rename_map[a] = "FTA"
    for a in ["freeThrowsPercentage", "FT%"]:
        if a in df.columns:
            rename_map[a] = "FT%"

    # Rebounds
    for a in ["offensiveRebounds", "OR", "OREB"]:
        if a in df.columns:
            rename_map[a] = "OR"
    for a in ["defensiveRebounds", "DR", "DREB"]:
        if a in df.columns:
            rename_map[a] = "DR"
    for a in ["totalRebounds", "REB", "TR", "TRB"]:
        if a in df.columns:
            rename_map[a] = "TR"

    # Points / playmaking / defense / fouls
    for a in ["pointsScored", "PTS", "Points"]:
        if a in df.columns:
            rename_map[a] = "PTS"
    for a in ["assists", "AST"]:
        if a in df.columns:
            rename_map[a] = "AST"
    for a in ["steals", "STL"]:
        if a in df.columns:
            rename_map[a] = "STL"
    for a in ["turnovers", "TOV", "TO"]:
        if a in df.columns:
            rename_map[a] = "TO"
    for a in ["blocks", "BLK"]:
        if a in df.columns:
            rename_map[a] = "BLK"
    for a in ["blocksAgainst", "BLKA", "BLK_AG"]:
        if a in df.columns:
            rename_map[a] = "BLKA"
    for a in ["foulsCommited", "foulsCommitted", "FC", "FLS_CM"]:
        if a in df.columns:
            rename_map[a] = "FC"
    for a in ["foulsDrawn", "FD", "FLS_RV"]:
        if a in df.columns:
            rename_map[a] = "FD"

    # Season / competition (τα κρατάμε κρυφά για join)
    for a in ["season", "Season"]:
        if a in df.columns:
            rename_map[a] = "season"
    for a in ["competition", "Competition"]:
        if a in df.columns:
            rename_map[a] = "competition"

    df = df.rename(columns=rename_map)

    for c in ["Player", "player_code", "Team"]:
        if c not in df.columns:
            df[c] = None

    if "TR" not in df.columns and all(c in df.columns for c in ["OR", "DR"]):
        df["TR"] = df["OR"].fillna(0) + df["DR"].fillna(0)

    target_cols = [
        "Player", "Team",
        "GP", "GS", "Min", "PTS",
        "2PM", "2PA", "2P%",
        "3PM", "3PA", "3P%",
        "FTM", "FTA", "FT%",
        "OR", "DR", "TR",
        "AST", "STL", "TO", "BLK", "BLKA",
        "FC", "FD", "PIR",
        "BCI", "Stability", "Form3",
    ]

    if "PIR" in df.columns:
        df = df.sort_values("PIR", ascending=False, na_position="last")

    keep = [c for c in target_cols if c in df.columns and c not in ["BCI", "Stability", "Form3"]]
    return df, keep, target_cols


def normalize_gamelogs_df(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for a in ["player_code", "code", "playerCode"]:
        if a in df.columns:
            ren[a] = "player_code"
    for a in ["player_name", "name", "playerName"]:
        if a in df.columns:
            ren[a] = "Player"
    for a in ["team_code", "teamCode", "Team"]:
        if a in df.columns:
            ren[a] = "Team"
    for a in ["opponent", "opponent_code", "Opp", "Opponent"]:
        if a in df.columns:
            ren[a] = "opponent"
    for a in ["game_date", "date", "gameDate", "Date"]:
        if a in df.columns:
            ren[a] = "game_date"
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
    made = made.astype(float).fillna(0)
    p = pct.astype(float).replace(0, np.nan)
    return (made / (p / 100.0)).round(1).fillna(0)

def add_feature_columns(players_df: pd.DataFrame) -> pd.DataFrame:
    df = players_df.copy()
    if "2PA" not in df.columns and all(c in df.columns for c in ["2PM", "2P%"]):
        df["2PA"] = compute_attempts_from_pct(df["2PM"], df["2P%"])
    if "3PA" not in df.columns and all(c in df.columns for c in ["3PM", "3P%"]):
        df["3PA"] = compute_attempts_from_pct(df["3PM"], df["3P%"])
    if "FTA" not in df.columns and all(c in df.columns for c in ["FTM", "FT%"]):
        df["FTA"] = compute_attempts_from_pct(df["FTM"], df["FT%"])

    df["FGA"] = df.get("2PA", 0).fillna(0) + df.get("3PA", 0).fillna(0)

    efg_num = df.get("2PM", 0).fillna(0) + 1.5 * df.get("3PM", 0).fillna(0)
    df["eFG%"] = (efg_num / df["FGA"].replace(0, np.nan)).fillna(0)

    denom = 2 * (df["FGA"] + 0.44 * df.get("FTA", 0).fillna(0))
    df["TS%"] = (df.get("PTS", 0).fillna(0) / denom.replace(0, np.nan)).fillna(0)

    df["FTR"] = (df.get("FTA", 0).fillna(0) / df["FGA"].replace(0, np.nan)).fillna(0)

    df["PTS/min"] = per_min(df.get("PTS", 0), df.get("Min", 0))
    df["TR/min"]  = per_min(df.get("TR", 0),  df.get("Min", 0))
    df["AST/min"] = per_min(df.get("AST", 0), df.get("Min", 0))
    df["FD/min"]  = per_min(df.get("FD", 0),  df.get("Min", 0))
    df["TO/min"]  = per_min(df.get("TO", 0),  df.get("Min", 0))
    df["STL/min"] = per_min(df.get("STL", 0), df.get("Min", 0))
    df["BLK/min"] = per_min(df.get("BLK", 0), df.get("Min", 0))
    df["Stocks/min"] = df["STL/min"] + df["BLK/min"]

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
        return pd.Series(0.5, index=s.index)
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
    if not mapping_text.strip():
        return df
    mp = {}
    for line in mapping_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 2:
            key, pos = parts[0], parts[1]
            mp[key] = pos

    if "player_code" in df.columns:
        df.loc[df["player_code"].astype(str).isin(mp.keys()), "Position"] = (
            df["player_code"].astype(str).map(mp)
        )
    if "Player" in df.columns:
        df.loc[df["Player"].astype(str).isin(mp.keys()), "Position"] = (
            df["Player"].astype(str).map(mp)
        )
    df["Position"] = df["Position"].apply(lambda x: str(x)[0].upper() if pd.notna(x) else x)
    return df

def position_score(df: pd.DataFrame, role: str) -> pd.Series:
    avail = (df.get("Min", 0).fillna(0) / 30.0).clip(0.6, 1.2)
    ts = df.get("TS%", 0).clip(0, 1)
    ef = df.get("eFG%", 0).clip(0, 1)
    twop = (df.get("2P%", 0) / 100.0).clip(0, 1)
    ftr = df.get("FTR", 0).clip(0, None)
    usage = minmax(df.get("Usage/min", 0))
    trm = minmax(df.get("TR/min", 0))
    astm = minmax(df.get("AST/min", 0))
    fdm = minmax(df.get("FD/min", 0))
    stocksm = minmax(df.get("Stocks/min", 0))
    tom = 1 - minmax(df.get("TO/min", 0))

    if role == "G":
        raw = (0.25*usage + 0.25*astm + 0.20*ts + 0.10*fdm + 0.10*ef + 0.05*stocksm + 0.05*tom)
    elif role == "F":
        raw = (0.25*trm + 0.20*usage + 0.15*ts + 0.10*ef + 0.10*fdm + 0.10*astm + 0.05*stocksm + 0.05*tom)
    else:
        raw = (0.30*trm + 0.20*stocksm + 0.15*twop + 0.15*fdm + 0.10*usage + 0.05*ts + 0.05*tom)

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

players_df_norm, base_cols, target_cols = normalize_players_df(players_df_raw)

gamelogs_df = None
if gamelogs_df_raw is not None and not gamelogs_df_raw.empty:
    gamelogs_df = normalize_gamelogs_df(gamelogs_df_raw.copy())

players_df_feat = add_feature_columns(players_df_norm)
players_df_feat["BCI"] = compute_bci(players_df_feat)

adv_sf = compute_stability_form3(players_df_feat, gamelogs_df)
players_df = players_df_feat.join(adv_sf)

players_df["All_Score_raw"] = universal_score_raw(players_df)
max_all = players_df["All_Score_raw"].max()
players_df["All_Score"] = (players_df["All_Score_raw"] / max_all * 100).round(1)

players_df["All_Score"] = universal_score(players_df)

form3_norm = scale_0_100_robust(players_df.get("Form3", pd.Series(index=players_df.index)))
stab_norm  = players_df.get("Stability", pd.Series(index=players_df.index)).astype(float)
players_df["PredictScore"] = (
    0.60*players_df["All_Score"].fillna(0) +
    0.30*form3_norm.fillna(50) +
    0.10*stab_norm.fillna(50)
) * (players_df.get("Min", 0).fillna(0)/30.0).clip(0.6, 1.15)
players_df["PredictScore"] = players_df["PredictScore"].clip(0, 100).round(1)

w_all, w_form3, w_stab = 0.60, 0.30, 0.10
base = (
    w_all   * players_df["All_Score"].astype(float).fillna(0) +
    w_form3 * form3_norm.fillna(50.0) +
    w_stab  * stab_norm.fillna(50.0)
)
min_mult = (players_df.get("Min", 0).fillna(0) / 30.0).clip(0.6, 1.15)
gp_mult  = np.where(players_df.get("GP", 0).fillna(0) >= 3, 1.0, 0.90)
players_df["PredictScore"] = (base * min_mult * gp_mult).clip(0, 100).round(1)

for _col in ["Stability", "Form3"]:
    if _col not in players_df.columns:
        players_df[_col] = np.nan

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
if "All_Score" in filtered_players.columns and "All_Score" not in final_cols:
    final_cols.append("All_Score")
if "PredictScore" in filtered_players.columns and "PredictScore" not in final_cols:
    final_cols.append("PredictScore")

st.subheader("Season Averages (με τις ζητούμενες στήλες + Advanced)")

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

table_df = filtered_players.copy()
if "Player" in table_df.columns and "player_code" in table_df.columns:
    table_df["Player"] = [
        _plink_player(c, n) for c, n in zip(table_df["player_code"], table_df["Player"])
    ]

display_cols = [c for c in final_cols if c in table_df.columns]

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

st.markdown(
    f"<div class='small-table'>{display_df.to_html(index=False, escape=False)}</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "📈 Player details (gamelogs)",
    "🧮 Advanced features",
    "🏆 Προτεινόμενα Picks (G/F/C)",
    "🔥 Top 30 (All)"
])

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
                with csum1:
                    st.metric("Games", len(player_gl))
                with csum2:
                    st.metric("PTS (avg)", round(player_gl.get("PTS", pd.Series([0])).mean(), 2))
                with csum3:
                    st.metric("PIR (avg)", round(player_gl.get("PIR", pd.Series([0])).mean(), 2))

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

with tabs[1]:
    st.markdown("### Advanced feature set (season-based)")
    feat_cols = [
        "Player", "Team", "Position", "Min", "PIR", "BCI",
        "TS%", "eFG%", "FTR",
        "Usage/min", "PTS/min", "TR/min", "AST/min", "FD/min", "Stocks/min", "TO/min",
        "Stability", "Form3", "All_Score", "PredictScore"
    ]
    feat_cols = [c for c in feat_cols if c in filtered_players.columns]
    # st.dataframe(filtered_players[feat_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

with tabs[2]:
    st.markdown("### Position mapping (optional)")
    st.caption(
        "Αν δεν υπάρχει στήλη θέσης στο CSV, μπορείς να δώσεις manual mapping σε μορφή `Player,POS` ή `player_code,POS` (POS ∈ G/F/C)."
        " Μία γραμμή ανά παίκτη. Αν δοθεί π.χ. `G/F`, λαμβάνεται το πρώτο γράμμα."
    )
    mapping_text = st.text_area("Μετατροπή", value="", height=120)

    players_pos = ensure_position_column(filtered_players.copy())
    if mapping_text.strip():
        players_pos = apply_manual_position_mapping(players_pos, mapping_text)

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


def gh_debug(owner: str, repo: str, token: str, workflow_filename: str):
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Check /user**")
        r = requests.get("https://api.github.com/user", headers=headers, timeout=15)
        st.code(f"GET /user → {r.status_code}")
        try:
            st.json(r.json())
        except Exception:
            st.write(r.text)

    with col2:
        st.write("**Check workflows list**")
        url_wf = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows"
        r2 = requests.get(url_wf, headers=headers, timeout=15)
        st.code(f"GET {url_wf} → {r2.status_code}")
        try:
            st.json(r2.json())
        except Exception:
            st.write(r2.text)

    st.write("**Try dispatch**")
    url_dispatch = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_filename}/dispatches"
    r3 = requests.post(url_dispatch, headers=headers, json={"ref": "main"}, timeout=20)
    st.code(f"POST {url_dispatch} → {r3.status_code}")
    st.write(r3.text if r3.text else "(no body)")
    if r3.status_code == 204:
        st.success("✅ Dispatch OK – δες το GitHub Actions.")
    elif r3.status_code == 401:
        st.error("❌ 401 Bad credentials: λάθος/μη εξουσιοδοτημένο token (scopes/SSO).")
    elif r3.status_code == 404:
        st.error("❌ 404: Μη προσβάσιμο repo/workflow με αυτό το token (fine-grained access ή λάθος owner/repo/filename).")

with st.expander("🧪 GitHub token debug"):
    owner = "N3rv0uS"  # π.χ. "N3rv0uS"
    repo  = "euroleague-fantasy-app"   # π.χ. "euroleague-fantasy-app"
    wf    = "euroleague_refresh.yml"
    token = st.secrets.get("GH_PAT", "")
    if st.button("Run GH debug"):
        if not token:
            st.error("Λείπει GH_PAT.")
        else:
            gh_debug(owner, repo, token, wf)
import requests

def gh_list_workflows(owner: str, repo: str, token: str):
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    r = requests.get(url, headers=headers, timeout=20)
    st.code(f"GET {url} -> {r.status_code}")
    data = r.json()
    # Δείξε όνομα & id για να αντιγράψεις
    items = []
    for wf in data.get("workflows", []):
        items.append({"name": wf.get("name"), "path": wf.get("path"), "id": wf.get("id")})
    st.write(items)

with st.expander("🧪 List GitHub workflows"):
    owner = "N3rv0uS"   # π.χ. N3rv0uS
    repo  = "euroleague-fantasy-app"    # π.χ. euroleague-fantasy-app
    token = st.secrets.get("GH_PAT", "")
    if st.button("List workflows"):
        gh_list_workflows(owner, repo, token)
