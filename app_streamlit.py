# app_streamlit.py
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import re
from urllib.parse import urlencode

# ---------- ΡΥΘΜΙΣΕΙΣ ΓΕΝΙΚΑ ----------
SEASON = "2025"  # ή E2025 αν έτσι δουλεύεις
OUT_DIR = Path("out")
avg_path = OUT_DIR / f"players_{SEASON}_perGame.csv"
urls_path = OUT_DIR / f"player_urls_{SEASON}.csv"

st.set_page_config(page_title="EuroLeague Fantasy – Player Game Logs", layout="wide")

# ---------- LOAD ΒΑΣΙΚΑ CSV ----------
df_avg = pd.read_csv(avg_path)
df_urls = pd.read_csv(urls_path)
df = df_avg.merge(df_urls[["player_code", "player_url", "Player"]].drop_duplicates("player_code"),
                  on="player_code", how="left")

# ---------- SCRAPER (SAFE) ----------
@st.cache_data(ttl=3600)
def scrape_gamelog_table(player_url: str):
    """
    Ασφαλές scraping: δοκιμάζει EL/EN & με/χωρίς slash.
    Διαβάζει ΜΟΝΟ <table> nodes. Επιστρέφει None αν δεν βρει πίνακα (δεν κάνει raise).
    """
    if not player_url:
        return None

    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception:
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
    variants.append(u.rstrip("/"))
    if not u.endswith("/"):
        variants.append(u + "/")
    if "/el/" in u:
        variants.append(u.replace("/el/", "/en/"))
    if "/en/" in u:
        variants.append(u.replace("/en/", "/el/"))

    def _pick_best_table(html: str) -> Optional[pd.DataFrame]:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        tables = []
        for t in soup.find_all("table"):
            try:
                part = pd.read_html(str(t))[0]  # ΔΙΑΒΑΖΕΙΣ ΜΟΝΟ ΤΟ εκάστοτε <table>
                tables.append(part)
            except Exception:
                continue
        if not tables:
            return None

        def score(df_tbl: pd.DataFrame) -> int:
            cols = [re.sub(r"\W+", "", str(c).lower()) for c in df_tbl.columns]
            keys = ["pir", "min", "λεπ", "pts", "πον", "date", "ημ", "opponent", "αντιπ"]
            return sum(any(k in c for c in cols) for k in keys)

        return max(tables, key=score)

    s = requests.Session()
    for v in variants:
        try:
            r = s.get(v, headers=headers, timeout=20, allow_redirects=True)
            best = _pick_best_table(r.text)
            if best is not None and not best.empty:
                return best
        except Exception:
            continue

    return None  # ποτέ raise

# ---------- ROUTER ΓΙΑ CLICK (?player_code=...) ----------
def show_player_page(player_code: str):
    """Δείξε σελίδα παίκτη (fallback σε scraping από official profile αν δεν υπάρχουν τοπικά gamelogs)."""
    pname = str(player_code)

    # Πάρε link & όνομα από το mapping CSV
    row = df[df["player_code"].astype(str) == str(player_code)].head(1)
    player_url = None
    if not row.empty:
        player_url = str(row.iloc[0].get("player_url", "")).strip()
        pname = row.iloc[0].get("Player", pname)

    st.title(f"{pname} — Αναλυτικά (Game-by-Game)")

    if not player_url:
        st.warning("Δεν βρέθηκε player_url για αυτόν τον παίκτη στο out/player_urls_2025.csv.")
        return

    gl = scrape_gamelog_table(player_url)
    if gl is None or gl.empty:
        st.warning("Δεν βρέθηκε HTML πίνακας με gamelogs στη σελίδα του παίκτη.")
        st.markdown(f"[Άνοιγμα επίσημου προφίλ]({player_url})")
        return

    st.dataframe(gl, use_container_width=True)
    for c in ["Πόντοι", "PTS", "PIR", "pir"]:
        if c in gl.columns:
            st.line_chart(gl[c])

# Αν υπάρχει ?player_code, δείξε σελίδα παίκτη και σταμάτα
pc = st.query_params.get("player_code")
if pc:
    show_player_page(pc)
    st.stop()

# ---------- ΥΠΟΛΟΙΠΗ ΛΟΓΙΚΗ (όπως είχες) ----------
# Helpers/metrics που είχες ήδη (δεν αλλάζω τη λογική σου)
def scale_0_100_robust(s: pd.Series, low_q=0.05, high_q=0.95) -> pd.Series:
    s = s.astype(float).replace([np.inf, -np.inf], np.nan)
    lo, hi = s.quantile(low_q), s.quantile(high_q)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(50.0, index=s.index)
    s_clip = s.clip(lo, hi)
    return ((s_clip - lo) / (hi - lo) * 100.0)

def per_min(x: pd.Series, minutes: pd.Series) -> pd.Series:
    x = x.fillna(0)
    m = minutes.replace(0, np.nan)
    return (x / m).fillna(0)

def compute_attempts_from_pct(made: pd.Series, pct: pd.Series) -> pd.Series:
    made = made.astype(float).fillna(0)
    p = pct.astype(float).replace(0, np.nan)
    return (made / (p / 100.0)).round(1).fillna(0)

def add_feature_columns(players_df: pd.DataFrame) -> pd.DataFrame:
    df2 = players_df.copy()

    # helper για ασφαλές get στήλης (επιστρέφει Series με 0 αν δεν υπάρχει)
    def safe_get(col, df=df2):
        return df[col] if col in df.columns else pd.Series(0, index=df.index)

    # attempts
    if "2PA" not in df2.columns and all(c in df2.columns for c in ["2PM", "2P%"]):
        made = safe_get("2PM").astype(float).fillna(0)
        pct = safe_get("2P%").astype(float).replace(0, np.nan)
        df2["2PA"] = (made / (pct / 100.0)).round(1).fillna(0)

    if "3PA" not in df2.columns and all(c in df2.columns for c in ["3PM", "3P%"]):
        made = safe_get("3PM").astype(float).fillna(0)
        pct = safe_get("3P%").astype(float).replace(0, np.nan)
        df2["3PA"] = (made / (pct / 100.0)).round(1).fillna(0)

    if "FTA" not in df2.columns and all(c in df2.columns for c in ["FTM", "FT%"]):
        made = safe_get("FTM").astype(float).fillna(0)
        pct = safe_get("FT%").astype(float).replace(0, np.nan)
        df2["FTA"] = (made / (pct / 100.0)).round(1).fillna(0)

    # totals
    df2["FGA"] = safe_get("2PA").fillna(0) + safe_get("3PA").fillna(0)

    # eFG%
    efg_num = safe_get("2PM").fillna(0) + 1.5 * safe_get("3PM").fillna(0)
    df2["eFG%"] = (efg_num / df2["FGA"].replace(0, np.nan)).fillna(0)

    # TS%
    denom = 2 * (df2["FGA"] + 0.44 * safe_get("FTA").fillna(0))
    df2["TS%"] = (safe_get("PTS").fillna(0) / denom.replace(0, np.nan)).fillna(0)

    # FTR
    df2["FTR"] = (safe_get("FTA").fillna(0) / df2["FGA"].replace(0, np.nan)).fillna(0)

    # per-minute metrics
    def per_min(x, minutes):
        x = x.fillna(0)
        m = minutes.replace(0, np.nan)
        return (x / m).fillna(0)

    df2["PTS/min"] = per_min(safe_get("PTS"), safe_get("Min"))
    df2["TR/min"]  = per_min(safe_get("TR"),  safe_get("Min"))
    df2["AST/min"] = per_min(safe_get("AST"), safe_get("Min"))
    df2["FD/min"]  = per_min(safe_get("FD"),  safe_get("Min"))
    df2["TO/min"]  = per_min(safe_get("TO"),  safe_get("Min"))
    df2["STL/min"] = per_min(safe_get("STL"), safe_get("Min"))
    df2["BLK/min"] = per_min(safe_get("BLK"), safe_get("Min"))
    df2["Stocks/min"] = df2["STL/min"] + df2["BLK/min"]

    # usage/min
    usage_numer = (
        safe_get("2PA").fillna(0)
        + safe_get("3PA").fillna(0)
        + 0.44 * safe_get("FTA").fillna(0)
        + safe_get("TO").fillna(0)
    )
    df2["Usage/min"] = per_min(usage_numer, safe_get("Min"))

    return df2


def compute_bci(players_df: pd.DataFrame) -> pd.Series:
    PTS_pm = players_df["PTS/min"]
    TR_pm  = players_df["TR/min"]
    AST_pm = players_df["AST/min"]
    FD_pm  = players_df["FD/min"]
    PIR_pm = per_min(players_df.get("PIR", 0), players_df.get("Min", 0))
    raw = 0.35*PTS_pm + 0.25*TR_pm + 0.25*AST_pm + 0.10*FD_pm + 0.05*PIR_pm
    min_bonus = (players_df.get("Min", 0).fillna(0) / 30.0).clip(0.6, 1.2)
    raw = raw * min_bonus
    return (100 * raw / raw.max()).round(1) if raw.max() > 0 else raw

def universal_score_raw(dfX: pd.DataFrame) -> pd.Series:
    avail = (dfX.get("Min", 0).fillna(0) / 30.0).clip(0.6, 1.2)
    def _minmax(s):
        s = s.replace([np.inf, -np.inf], np.nan)
        if s.max(skipna=True) == s.min(skipna=True):
            return pd.Series(0.5, index=s.index)
        return (s - s.min(skipna=True)) / (s.max(skipna=True) - s.min(skipna=True))
    ts       = dfX.get("TS%", 0).clip(0, 1)
    usage    = _minmax(dfX.get("Usage/min", 0))
    trm      = _minmax(dfX.get("TR/min", 0))
    astm     = _minmax(dfX.get("AST/min", 0))
    fdm      = _minmax(dfX.get("FD/min", 0))
    stocksm  = _minmax(dfX.get("Stocks/min", 0))
    tom_good = 1 - _minmax(dfX.get("TO/min", 0))
    raw = 0.20*usage + 0.18*ts + 0.18*trm + 0.16*astm + 0.12*fdm + 0.10*stocksm + 0.06*tom_good
    return raw * avail

def universal_score(dfX: pd.DataFrame) -> pd.Series:
    score = universal_score_raw(dfX)
    return (100 * score / score.max()).round(1) if score.max() > 0 else score

def compute_stability_form3(players_df: pd.DataFrame, gl_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    adv = pd.DataFrame(index=players_df.index)
    stability_map, form3_map = {}, {}
    if gl_df is not None and not gl_df.empty:
        key = "player_code" if "player_code" in gl_df.columns else ("Player" if "Player" in gl_df.columns else None)
        if key is not None:
            for pid, g in gl_df.groupby(key):
                g = g.sort_values("game_date") if "game_date" in g.columns else g
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

# ---------- ΕΜΠΛΟΥΤΙΣΜΟΣ & SCORES ----------
players_df_feat = add_feature_columns(df.copy())
players_df_feat["BCI"] = compute_bci(players_df_feat)

# Δεν φορτώνουμε gamelogs εδώ (η σελίδα παίκτη τα φέρνει αν υπάρχουν μέσω scraping)
adv_sf = compute_stability_form3(players_df_feat, None)
players_df_all = players_df_feat.join(adv_sf)

players_df_all["All_Score_raw"] = universal_score_raw(players_df_all)
max_all = players_df_all["All_Score_raw"].max()
players_df_all["All_Score"] = (players_df_all["All_Score_raw"] / max_all * 100).round(1) if max_all and max_all > 0 else 0
players_df_all["All_Score"] = universal_score(players_df_all)

form3_norm = scale_0_100_robust(players_df_all.get("Form3", pd.Series(index=players_df_all.index)))
stab_norm  = players_df_all.get("Stability", pd.Series(index=players_df_all.index)).astype(float)
w_all, w_form3, w_stab = 0.60, 0.30, 0.10
base = (w_all*players_df_all["All_Score"].fillna(0) + w_form3*form3_norm.fillna(50) + w_stab*stab_norm.fillna(50))
min_mult = (players_df_all.get("Min", 0).fillna(0) / 30.0).clip(0.6, 1.15)
gp_mult  = np.where(players_df_all.get("GP", 0).fillna(0) >= 3, 1.0, 0.90)
players_df_all["PredictScore"] = (base * min_mult * gp_mult).clip(0, 100).round(1)

# ---------- ΦΙΛΤΡΑ ----------
teams_list = ["(Όλες)"] + sorted(players_df_all.get("Team", pd.Series(dtype=str)).dropna().astype(str).unique(), key=lambda x: x.lower())
f1, f2, f3 = st.columns([2, 1, 1])
with f1:
    q = st.text_input("🔎 Live search (όνομα/κωδικός/ομάδα)", "")
with f2:
    team_sel = st.selectbox("Ομάδα", teams_list, index=0)
with f3:
    min_gp = st.number_input("Min GP", min_value=0, max_value=50, value=0, step=1)

def filter_players(dfF: pd.DataFrame, qtxt: str, team: str, min_gp_v: int) -> pd.DataFrame:
    res = dfF.copy()
    if qtxt:
        qlow = qtxt.lower().strip()
        res = res[
            res.get("Player", pd.Series(index=res.index, dtype=str)).fillna("").str.lower().str.contains(qlow)
            | res.get("player_code", pd.Series(index=res.index, dtype=str)).fillna("").astype(str).str.contains(qlow)
            | res.get("Team", pd.Series(index=res.index, dtype=str)).fillna("").str.lower().str.contains(qlow)
        ]
    if team and team != "(Όλες)":
        res = res[res.get("Team", pd.Series(index=res.index, dtype=str)).fillna("").str.lower().eq(team.lower())]
    if "GP" in res.columns and min_gp_v > 0:
        res = res[res["GP"].fillna(0) >= min_gp_v]
    return res

filtered_players = filter_players(players_df_all, q, team_sel, min_gp)

# ---------- ΣΤΗΛΕΣ ΠΙΝΑΚΑ ----------
target_cols = [
    "Player", "Team", "GP", "GS", "Min", "PTS",
    "2PM", "2PA", "2P%", "3PM", "3PA", "3P%", "FTM", "FTA", "FT%",
    "OR", "DR", "TR", "AST", "STL", "TO", "BLK", "BLKA", "FC", "FD", "PIR",
    "BCI", "Stability", "Form3", "All_Score", "PredictScore"
]
final_cols = [c for c in target_cols if c in filtered_players.columns]

if "All_Score" in filtered_players.columns and "All_Score" not in final_cols:
    final_cols.append("All_Score")
if "PredictScore" in filtered_players.columns and "PredictScore" not in final_cols:
    final_cols.append("PredictScore")

# ---------- RENDER SEASON AVERAGES ΠΙΝΑΚΑΣ ----------
st.subheader("Season Averages (με τις ζητούμενες στήλες + Advanced)")

# μικρότερο font & padding
st.markdown("""
<style>
.small-table table { font-size: 14px; }
.small-table th, .small-table td { padding: 6px 10px; }
.small-table a { color: inherit; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# Κάνε τη στήλη Player clickable (χρειάζεται player_code)
if "Player" in filtered_players.columns and "player_code" in filtered_players.columns:
    table_df = filtered_players.copy()
    table_df["Player"] = [
        f'<a href="?{urlencode({"player_code": str(code)})}">{name}</a>'
        for code, name in zip(table_df["player_code"], table_df["Player"])
    ]
else:
    table_df = filtered_players.copy()

display_cols = [c for c in final_cols if c in table_df.columns]

# Show more / less κάτω από τον πίνακα
is_all = st.session_state.get("show_all", False)
disp = table_df[display_cols].reset_index(drop=True)
if not is_all:
    disp = disp.head(30)

st.markdown(
    f"<div class='small-table'>{disp.to_html(index=False, escape=False)}</div>",
    unsafe_allow_html=True,
)

if not is_all:
    if st.button("Show more"):
        st.session_state["show_all"] = True
        st.rerun()
else:
    if st.button("Show less"):
        st.session_state["show_all"] = False
        st.rerun()

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
        # εδώ, αν έχεις τοπικά gamelogs φορτωμένα, μπορείς να τα δείξεις αντί για scraping
        st.info("Tip: Για 100% σταθερότητα, προτίμησε τα τοπικά gamelogs (Schedule→Boxscore) όταν ετοιμαστούν.")

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
    # (προβολή αν τη χρειαστείς)
    # st.dataframe(filtered_players[feat_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

# --- TAB 3: Position-aware Picks ---
# (κρατάω το δικό σου κομμάτι όπως το είχες – αν θες επαναφέρω όλο το mapping/picks)

# --- TAB 4: Top 30 (All) ---
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
