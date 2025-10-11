# fetch_euroleague_stats.py
import argparse
import os
import time
import sys
import csv
from typing import Iterable, List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re, requests, pandas as pd
import requests
import pandas as pd
import streamlit as st
from urllib.parse import urlencode
import inspect


BASE_URL = "https://feeds.incrowdsports.com/provider/euroleague-feeds/v3/competitions/{competition}/statistics/players/traditional"
# πάνω-πάνω, global
_SESSION = requests.Session()
DEFAULT_HEADERS = {
    # πρόσθεσε headers αν ποτέ χρειαστεί (π.χ. User-Agent)
    "Accept": "application/json, text/plain, */*",
}
# -------- robust helpers for Incrowd variants --------
# ----------------- CMS content gamelogs parser (no scraping) -----------------



def _dig(d, path, default=None):
    cur = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, list):
            try:
                i = int(p)
                cur = cur[i]
            except:
                return default
        else:
            return default
    return cur

def _try_json(url: str, params: Optional[dict] = None, timeout: int = 30) -> dict:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def parse_player_gamelogs_from_content(content_json: dict, player_code: Optional[str] = None) -> pd.DataFrame:
    """
    Παρσάρει τον πίνακα Game stats από το CMS JSON που φορτώνει το frontend.
    Περιμένει δομή τύπου:
      stats.currentSeason.gameStats[0].table.sections  (ανά κατηγορία)
      stats.currentSeason.gameStats[0].table.headSection.stats  (meta ανά παιχνίδι: gameUrl, αντίπαλος κ.λπ.)

    Επιστρέφει DF με ένα row ανά παιχνίδι και στήλες: MIN, PTS, 2FG, 3FG, FT, O, D, T, AST, STL, TOV,
    BLK_FV, BLK_AG, FLS_CM, FLS_RV, PIR + game_url/opponent/home_away/game_date, player_code, game_no.
    """
    node = _dig(content_json, "stats.currentSeason.gameStats.0.table")
    if node is None:
        node = _dig(content_json, "gameStats.0.table")
    if node is None:
        return pd.DataFrame()

    head_stats = _dig(node, "headSection.stats", [])
    sections = node.get("sections", [])

    sec_frames: List[pd.DataFrame] = []
    for sec in sections:
        rows = sec.get("rows") or sec.get("stats") or []
        # header inference
        header = []
        for row in rows[:1]:
            if isinstance(row, list):
                for cell in row:
                    if isinstance(cell, dict):
                        k = cell.get("statType") or cell.get("label") or cell.get("type") or ""
                        header.append(k)
        if not header:
            header = [c.get("statType") or c.get("key") or c.get("name") for c in sec.get("columns", [])]
        data = []
        for row in rows:
            if isinstance(row, list):
                data.append([cell.get("value") if isinstance(cell, dict) else cell for cell in row])
        df = pd.DataFrame(data, columns=header if header and data else None)
        df.insert(0, "__idx__", range(1, len(df) + 1))
        df = df.loc[:, [c for c in df.columns if c == "__idx__" or (isinstance(c, str) and c)]]
        sec_frames.append(df)

    body = pd.DataFrame()
    if sec_frames:
        body = sec_frames[0]
        for df in sec_frames[1:]:
            body = body.merge(df, on="__idx__", how="outer")

    meta_rows = []
    for i, s in enumerate(head_stats, start=1):
        if isinstance(s, dict):
            game_url = s.get("gameUrl") or s.get("url") or s.get("href")
            opp = s.get("opponent") or s.get("opponentName") or s.get("name")
            home_away = s.get("homeAway") or s.get("location")
            game_date = s.get("gameDate") or s.get("date")
        else:
            game_url = opp = home_away = game_date = None
        meta_rows.append({
            "__idx__": i,
            "game_url": game_url, "opponent": opp, "home_away": home_away, "game_date": game_date
        })
    meta = pd.DataFrame(meta_rows)

    out = meta.merge(body, on="__idx__", how="outer").sort_values("__idx__")
    out["player_code"] = player_code
    out["game_no"] = out["__idx__"]
    out.drop(columns=["__idx__"], inplace=True, errors="ignore")

    alias_map = {
        "MIN": ["min","minutes","MIN","time","minutesPlayed"],
        "PTS": ["pts","points","PTS","pointsScored"],
        "2FG": ["2fg","twoPoints","twoPointers","twoPointersMadeAttempted","twoPointersShort","twoPointersText"],
        "3FG": ["3fg","threePoints","threePointers","threePointersMadeAttempted","threePointersShort","threePointersText"],
        "FT":  ["ft","freeThrows","freeThrowsMadeAttempted","freeThrowsShort","freeThrowsText"],
        "O":   ["o","off","offensive","offensiveRebounds","reb_o","O"],
        "D":   ["d","def","defensive","defensiveRebounds","reb_d","D"],
        "T":   ["t","tot","total","totalRebounds","reb_t","T"],
        "AST": ["as","ast","assists","AST"],
        "STL": ["st","steals","STL"],
        "TOV": ["to","turnovers","TOV"],
        "BLK_FV": ["fv","blocksFor","blocksMade","blocks"],
        "BLK_AG": ["ag","blocksAgainst","blocksReceived"],
        "FLS_CM": ["cm","foulsCommitted","foulsCommited","foulsCommittedValue"],
        "FLS_RV": ["rv","foulsDrawn","foulsReceived"],
        "PIR": ["pir","indexRating","PIR"],
    }

    # rename όπου ταιριάζει
    std_cols = {k: None for k in alias_map}
    for std, keys in alias_map.items():
        for k in list(out.columns):
            if str(k).strip() in keys:
                std_cols[std] = k
                break
    rename = {v: k for k,v in std_cols.items() if v}
    out = out.rename(columns=rename)

    # split made/attempts για 2FG/3FG/FT, αν είναι "x/y"
    def split_made_att(s):
        if isinstance(s, str) and "/" in s:
            a,b = s.split("/", 1)
            try: return float(a), float(b)
            except: return a,b
        return None,None

    for cat in ["2FG","3FG","FT"]:
        if cat in out.columns:
            m,a = zip(*[split_made_att(x) for x in out[cat].tolist()])
            out[f"{cat}_M"] = m; out[f"{cat}_A"] = a

    # βεβαιώσου ότι υπάρχουν όλα τα standard cols
    for c in alias_map.keys():
        if c not in out.columns:
            out[c] = pd.NA

    return out

def _json_rows(payload):
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("items", "data", "rows", "list"):
            v = payload.get(k)
            if isinstance(v, list):
                return v
        # first list value
        for v in payload.values():
            if isinstance(v, list):
                return v
    return []

def _probe(urls: List[str], param_variants: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Δοκιμάζει διαδοχικά endpoints & σετ παραμέτρων. Επιστρέφει το πρώτο DF που έχει rows.
    Γράφει μικρά logs για να ξέρεις τι «έπιασε».
    """
    for u in urls:
        for params in param_variants:
            try:
                payload = _request_json(u, params)
                rows = _json_rows(payload)
                if rows:
                    df = pd.json_normalize(rows, sep="_")
                    print(f"[ok] gamelogs via {u} with { {k:params[k] for k in sorted(params)} }  rows={len(df)}")
                    return df
                else:
                    print(f"[info] empty via {u} with {params}")
            except Exception as e:
                print(f"[info] 400/err via {u} with {params}: {e}")
    return pd.DataFrame()

def fetch_player_games(
    player_code: str,
    season: str,
    competition: str = "E",
    mode: str = "perGame",
    content_url: Optional[str] = None,
    content_json_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Παίρνει gamelogs για ΕΝΑΝ παίκτη από το CMS JSON.
    - content_url: πλήρες URL του JSON (ή template με {player_code} / {season})
    - content_json_path: αν έχεις το JSON αποθηκευμένο τοπικά (για δοκιμή)

    Προτεραιότητα: content_json_path > content_url
    """
    import json, os
    payload = None

    if content_json_path and os.path.exists(content_json_path):
        with open(content_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    elif content_url:
        url = content_url.format(player_code=player_code, season=season, competition=competition)
        payload = _try_json(url)
    else:
        raise ValueError("Provide either content_url or content_json_path")

    df = parse_player_gamelogs_from_content(payload, player_code=player_code)
    df["season"] = season
    df["competition"] = competition
    df["mode"] = mode
    return df

def fetch_all_gamelogs_single_call(
    season: str,
    competition: str = "E",
    mode: str = "perGame",
    limit: int = 100000,
) -> pd.DataFrame:
    """
    Προσπαθεί να φέρει ΟΛΑ τα player-game rows με ΕΝΑ call.
    Σε πολλά Incrowd setups υπάρχει το pattern:
      /statistics/games/players/traditional
    Αν δεν επιστρέψει τίποτα, γυρνάει κενό DF (και θα πέσουμε σε fallback).
    """
    url = f"https://feeds.incrowdsports.com/provider/euroleague-feeds/v3/competitions/{competition}/statistics/games/players/traditional"
    params = {
      "seasonMode": "Range",
      "fromSeasonCode": _season_code(competition, season),
      "toSeasonCode": _season_code(competition, season),
      "statisticMode": mode,
      "limit": limit,
      # "statisticSortMode": "GameDate",  # optional
    }
    try:
        payload = _request_json(url, params)
        df = _json_to_df(payload)
        if not df.empty:
            df["season"] = season
            df["competition"] = competition
            df["mode"] = mode
        return df
    except Exception as e:
        print(f"[info] single-call gamelogs endpoint not available or failed: {e}", file=sys.stderr)
        return pd.DataFrame()
# dummy commit to refresh GitHub Actions

def fetch_all_player_gamelogs(
    season: str,
    competition: str = "E",
    mode: str = "perGame",
    player_codes: Optional[Iterable[str]] = None,
    content_url_template: Optional[str] = None,
    content_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Μαζεύει gamelogs για όλους:
    - Αν δώσεις content_url_template → θα καλέσει URL ανά παίκτη (format με {player_code}, {season}, {competition})
    - Αλλιώς αν δώσεις content_dir → ψάχνει αρχεία JSON: {content_dir}/{player_code}.json
    """
    import os, pandas as pd

    # 1) φτιάξε λίστα παικτών
    codes: List[str] = []
    if player_codes:
        codes = [str(x).strip() for x in player_codes if str(x).strip()]
    else:
        master_path = os.path.join("out", f"players_{season}_{mode}.csv")
        if os.path.exists(master_path):
            m = pd.read_csv(master_path)
        else:
            m = fetch_season_averages(season=season, competition=competition, mode=mode, limit=1000)
        code_col = None
        for c in m.columns:
            if c in ("player_code", "code") or str(c).endswith(".code") or str(c).endswith("_code"):
                code_col = c; break
        if not code_col:
            raise RuntimeError("Could not locate player_code column in master feed.")
        codes = m[code_col].dropna().astype(str).unique().tolist()

    frames = []
    for code in codes:
        try:
            if content_dir:
                p = os.path.join(content_dir, f"{code}.json")
                df = fetch_player_games(code, season, competition, mode, content_json_path=p)
            else:
                df = fetch_player_games(code, season, competition, mode, content_url=content_url_template)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"[warn] gamelogs failed for player_code={code}: {e}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


    def _one(pcode: str) -> pd.DataFrame:
        return fetch_player_games(player_code=pcode, season=season, competition=competition, mode=mode)

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one, p): p for p in player_codes}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                dfi = fut.result()
                if not dfi.empty:
                    frames.append(dfi)
            except Exception as e:
                print(f"[warn] gamelog failed for {p}: {e}", file=sys.stderr)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"[progress] {done}/{total}")

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for col, val in [("season", season), ("competition", competition), ("mode", mode)]:
        if col not in out.columns:
            out[col] = val
    print(f"[info] Fallback collected rows={len(out)}")
    return out


    def _one(code: str) -> pd.DataFrame:
        df_i = fetch_player_games(player_code=code, season=season, competition=competition, mode=mode)
        return df_i

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_one, p): p for p in player_codes}
        done = 0
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                df_i = fut.result()
                if not df_i.empty:
                    frames.append(df_i)
            except Exception as e:
                print(f"[warn] Fail for player_code={p}: {e}", file=sys.stderr)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"[progress] {done}/{total} players")

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for col, val in [("season", season), ("competition", competition), ("mode", mode)]:
        if col not in out.columns:
            out[col] = val
    return out


def _request_json(url: str, params: Dict[str, Any], max_retries: int = 2, timeout: float = 15.0, sleep_sec: float = 1.0) -> Dict[str, Any]:
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = _SESSION.get(url, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
            # 429: rate limit → κάνε ευγενική παύση & ξαναπροσπάθησε
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "20"))
                time.sleep(max(retry_after, 20))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(sleep_sec * attempt)  # simple backoff
            else:
                raise
    raise last_exc if last_exc else RuntimeError("Unknown request error")

# ---------- helpers ----------

def _season_code(competition: str, season: str) -> str:
    # E2025, U2025 κ.λπ.
    return f"{competition}{season}"

def _request_json(url: str, params: Dict[str, Any], max_retries: int = 3, sleep_sec: float = 1.0) -> Dict[str, Any]:
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(sleep_sec * attempt)  # απλό backoff
            else:
                raise
    # safety
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown request error")

def _json_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Το Incrowd επιστρέφει συνήθως λίστα αντικειμένων κάτω από keys όπως 'items' ή 'data'.
    Καλύπτουμε και τα δύο σενάρια προσεκτικά.
    """
    if isinstance(payload, list):
        data = payload
    elif isinstance(payload, dict):
        if "items" in payload and isinstance(payload["items"], list):
            data = payload["items"]
        elif "data" in payload and isinstance(payload["data"], list):
            data = payload["data"]
        else:
            # προσπάθησε να βρεις πρώτη list value
            data = None
            for v in payload.values():
                if isinstance(v, list):
                    data = v
                    break
            if data is None:
                # τελευταίο fallback: τύλιξέ το
                data = [payload]
    else:
        data = [payload]
    df = pd.json_normalize(data, sep="_")
    return df

def _ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

# ---------- βασικές συναρτήσεις (season averages που ήδη έχεις) ----------

def fetch_season_averages(season: str, competition: str = "E", mode: str = "perGame", limit: int = 1000) -> pd.DataFrame:
    """
    Κατεβάζει τα season averages (perGame) από το Incrowd feed.
    """
    url = BASE_URL.format(competition=competition)
    params = {
        "seasonMode": "Range",
        "fromSeasonCode": _season_code(competition, season),
        "toSeasonCode": _season_code(competition, season),
        "statisticMode": mode,
        "limit": limit,
    }
    payload = _request_json(url, params)
    df = _json_to_df(payload)
    # πρόσθετες στήλες ταυτότητας
    df["season"] = season
    df["competition"] = competition
    df["mode"] = mode
    return df

# ---------- ΝΕΟ: gamelogs για έναν παίκτη ----------

def fetch_player_games(
    player_code: str,
    season: str,
    competition: str = "E",
    mode: str = "perGame",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Επιστρέφει αναλυτικά stats ανά αγώνα για συγκεκριμένο παίκτη (ένα row ανά game).
    Χρησιμοποιεί το ίδιο 'traditional' feed με παραμέτρους:
      - statisticSortMode=GameDate
      - playerCodes=<PLAYER_CODE>
    """
    url = BASE_URL.format(competition=competition)
    params = {
        "seasonMode": "Range",
        "fromSeasonCode": _season_code(competition, season),
        "toSeasonCode": _season_code(competition, season),
        "statisticMode": mode,
        "statisticSortMode": "GameDate",
        "playerCodes": player_code,
        "limit": limit,
    }
    payload = _request_json(url, params)
    df = _json_to_df(payload)

    # Βάζουμε μεταδεδομένα & κανονικοποιούμε πιθανά nested fields
    df["player_code"] = player_code
    df["season"] = season
    df["competition"] = competition
    df["mode"] = mode

    # Μικρές διευθετήσεις (αν υπάρχουν κοινά fields):
    # - Από το feed συνήθως υπάρχουν game identifiers & ημ/νίες π.χ. 'game_gameDate' ή 'game_date'
    # Δεν γνωρίζουμε ακριβή schema σου — κρατάμε ό,τι έρχεται.
    # Αν θες ρητές στήλες, κάνε rename εδώ, π.χ.:
    # mapping = {"game_gameDate": "gameDate", "team_shortName": "teamShort"}
    # df = df.rename(columns={k:v for k,v in mapping.items() if k in df.columns})

    return df

# ---------- ΝΕΟ: gamelogs για όλους ----------

def fetch_all_player_gamelogs(
    season: str,
    competition: str = "E",
    mode: str = "perGame",
    player_codes: Optional[Iterable[str]] = None,
    master_limit: int = 1000,
) -> pd.DataFrame:
    """
    Αν δεν δοθούν player_codes:
      - πρώτα φέρνουμε το master season feed για να αντλήσουμε όλους τους player.code
    Μετά κάνουμε loop & concat τα gamelogs.
    """
    if player_codes is None:
        master_df = fetch_season_averages(season=season, competition=competition, mode=mode, limit=master_limit)
        # Το πεδίο του κωδικού συνήθως είναι 'player_code' ή 'player.code' (json_normalize → 'player_code')
        candidate_cols = [c for c in master_df.columns if c.endswith("player.code") or c.endswith("player_code")]
        if not candidate_cols:
            # ψάξε και για 'code'
            candidate_cols = [c for c in master_df.columns if c.endswith("_code") or c == "code"]
        if not candidate_cols:
            raise ValueError("Δεν βρέθηκε στήλη με player code στο master feed. Έλεγξε τα columns του season averages.")

        code_col = candidate_cols[0]
        player_codes = (
            master_df[[code_col]]
            .dropna()
            .drop_duplicates()
            .astype(str)[code_col]
            .tolist()
        )

    frames: List[pd.DataFrame] = []
    for i, pcode in enumerate(player_codes, start=1):
        try:
            df_i = fetch_player_games(player_code=pcode, season=season, competition=competition, mode=mode)
            if not df_i.empty:
                frames.append(df_i)
        except Exception as e:
            # Μην «σπάει» το batch για 1 αποτυχία — απλά γράψε stderr
            print(f"[warn] Απέτυχαν gamelogs για player_code={pcode}: {e}", file=sys.stderr)
        # μικρό ρυθμιστικό διάλειμμα για να είμαστε ευγενικοί με το feed
        time.sleep(0.15)

    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame()

    # Βεβαιώσου ότι υπάρχουν τα βασικά metadata (σε περίπτωση κενών)
    for col, val in [("season", season), ("competition", competition), ("mode", mode)]:
        if col not in out.columns:
            out[col] = val

    return out

# ---------- αποθήκευση ----------

def save_csv(df: pd.DataFrame, path: str) -> None:
    _ensure_out_dir(os.path.dirname(path) or ".")
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)

# ---------- CLI ----------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EuroLeague/EuroCup stats fetcher (Incrowd feeds)")
    p.add_argument("--kind", choices=["season", "gamelogs"], required=True, help="Τύπος λήψης: season averages ή game-by-game")
    p.add_argument("--seasons", required=True, help="Π.χ. 2025 ή 2024,2025")
    p.add_argument("--competition", default="E", help="E (EuroLeague) ή U (EuroCup)")
    p.add_argument("--mode", default="perGame", help="statisticMode, π.χ. perGame")
    p.add_argument("--out", default="out/", help="Φάκελος εξαγωγής")
    p.add_argument("--players", default="", help="Συγκεκριμένοι player codes χωρισμένοι με κόμμα π.χ. 002661,011196")
    p.add_argument("--limit", type=int, default=1000, help="limit παραμέτρου για feed")
    p.add_argument("--content-url", default="", help="CMS JSON URL template (use {player_code},{season},{competition})")
    p.add_argument("--content-dir", default="", help="Folder with CMS JSON dumps named <player_code>.json")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # seasons & players parsing
    seasons = [s.strip() for s in str(args.seasons).split(",") if s.strip()]
    players_list = [p.strip() for p in str(args.players).split(",") if p.strip()] if args.players else None

    for season in seasons:
        if args.kind == "season":
            df = fetch_season_averages(
                season=season,
                competition=args.competition,
                mode=args.mode,
                limit=args.limit,
            )
            out_path = os.path.join(args.out, f"players_{season}_{args.mode}.csv")
            save_csv(df, out_path)
            print(f"[ok] Saved: {out_path} (rows={len(df)})")

        elif args.kind == "gamelogs":
            print(f"[info] Start gamelogs for season={season}, competition={args.competition}, mode={args.mode}")
            print("fetch_all_player_gamelogs params:", list(inspect.signature(fetch_all_player_gamelogs).parameters))

            df = fetch_all_player_gamelogs(
                season=season,
                competition=args.competition,
                mode=args.mode,
                player_codes=players_list,
                content_url_template=(args.content_url or None),
                content_dir=(args.content_dir or None),
            )
            out_path = os.path.join(args.out, f"player_gamelogs_{season}_{args.mode}.csv")
            save_csv(df, out_path)
            print(f"[ok] Saved: {out_path} (rows={len(df)})")


        else:
            raise ValueError(f"Unknown kind: {args.kind}")


if __name__ == "__main__":
    main()


