# fetch_euroleague_stats.py
import argparse
import os
import time
import sys
import csv
from typing import Iterable, List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd

BASE_URL = "https://feeds.incrowdsports.com/provider/euroleague-feeds/v3/competitions/{competition}/statistics/players/traditional"
# πάνω-πάνω, global
_SESSION = requests.Session()
DEFAULT_HEADERS = {
    # πρόσθεσε headers αν ποτέ χρειαστεί (π.χ. User-Agent)
    "Accept": "application/json, text/plain, */*",
}
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

def fetch_all_player_gamelogs(
    season: str,
    competition: str = "E",
    mode: str = "perGame",
    player_codes: Optional[Iterable[str]] = None,
    master_limit: int = 1000,
    max_workers: int = 8,
    only_active: bool = True,
) -> pd.DataFrame:
    """
    1) Πρώτα δοκιμάζουμε single-call (games→players).
    2) Αν βγει κενό, fallback: loop σε playerCodes (παλιή μέθοδος).
    """

    # --- 1) Single-call attempt ---
    print("[info] Trying single-call gamelogs ...")
    df_single = fetch_all_gamelogs_single_call(season, competition, mode)
    if not df_single.empty:
        print(f"[ok] Single-call returned rows={len(df_single)}")
        return df_single

    print("[warn] Single-call returned empty. Falling back to per-player loop.")

    # --- 2) Fallback: per-player loop ---
    if player_codes is None:
        master_df = fetch_season_averages(season=season, competition=competition, mode=mode, limit=master_limit)

        # Στήλη με player_code: στο δικό σου CSV είναι *ακριβώς* "player_code"
        if "player_code" in master_df.columns:
            code_col = "player_code"
        else:
            cand = [c for c in master_df.columns if c.endswith("player.code") or c.endswith("player_code") or c == "code"]
            if not cand:
                raise ValueError("Δεν βρέθηκε στήλη με player code στο master feed.")
            code_col = cand[0]

        if only_active:
            # gamesPlayed υπάρχει στα season averages που έδειξες
            gp_candidates = [c for c in ["gamesPlayed","GP","games","statistics_games"] if c in master_df.columns]
            if gp_candidates:
                gp_col = gp_candidates[0]
                master_df = master_df[pd.to_numeric(master_df[gp_col], errors="coerce").fillna(0) > 0]

        player_codes = (
            master_df[[code_col]].dropna().drop_duplicates().astype(str)[code_col].tolist()
        )

    from concurrent.futures import ThreadPoolExecutor, as_completed
    frames: List[pd.DataFrame] = []
    total = len(player_codes)
    print(f"[info] Fallback: fetching gamelogs per player for {total} players ...")

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
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    players_list = [p.strip() for p in args.players.split(",") if p.strip()] if args.players else None

    for season in seasons:
         if args.kind == "season":
            df = fetch_season_averages(season=season, competition=args.competition, mode=args.mode, limit=args.limit)
            out_path = os.path.join(args.out, f"players_{season}_{args.mode}.csv")
            save_csv(df, out_path)
            print(f"[ok] Saved: {out_path} (rows={len(df)})")

         elif args.kind == "gamelogs":
            print(f"[info] Start gamelogs for season={season}, competition={args.competition}, mode={args.mode}")
            df = fetch_all_player_gamelogs(
            season=season,
            competition=args.competition,
            mode=args.mode,
            player_codes=players_list,
            master_limit=args.limit,
            )
    if df.empty:
        print("[warn] gamelogs came back EMPTY. Check endpoint availability and params.", file=sys.stderr)
    out_path = os.path.join(args.out, f"player_gamelogs_{season}_{args.mode}.csv")
    save_csv(df, out_path)
    print(f"[ok] Saved: {out_path} (rows={len(df)})")


if __name__ == "__main__":
    main()
