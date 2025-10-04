#!/usr/bin/env python3
import argparse, json, os, sys, requests
import pandas as pd

def _dig(d, path, default=None):
    cur = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, list):
            try:
                cur = cur[int(p)]
            except Exception:
                return default
        else:
            return default
    return cur

def parse_player_gamelogs_from_content(content_json: dict, player_code: str|None=None) -> pd.DataFrame:
    # βρίσκουμε τον πίνακα "Game stats"
    table = _dig(content_json, "stats.currentSeason.gameStats.0.table") \
         or _dig(content_json, "gameStats.0.table")
    if not table:
        return pd.DataFrame()

    head_stats = _dig(table, "headSection.stats", []) or []
    sections = table.get("sections", [])

    # φτιάχνουμε dataframes ανά section και τα ενώνουμε με σειρά (__idx__)
    sec_frames = []
    for sec in sections:
        rows = sec.get("rows") or sec.get("stats") or []
        # header (προσπαθούμε να βγάλουμε ονόματα στηλών)
        header = []
        if rows and isinstance(rows[0], list):
            for cell in rows[0]:
                if isinstance(cell, dict):
                    header.append(cell.get("statType") or cell.get("label") or cell.get("type") or "")
                else:
                    header.append(str(cell))
        data = []
        for row in rows:
            if isinstance(row, list):
                data.append([c.get("value") if isinstance(c, dict) else c for c in row])
        df = pd.DataFrame(data, columns=header if header and len(header)==len(data[0]) else None)
        df.insert(0, "__idx__", range(1, len(df)+1))
        sec_frames.append(df)

    body = pd.DataFrame()
    if sec_frames:
        body = sec_frames[0]
        for df in sec_frames[1:]:
            body = body.merge(df, on="__idx__", how="outer")

    # meta ανά παιχνίδι (gameUrl, opponent, home/away, date)
    meta_rows = []
    for i, s in enumerate(head_stats, start=1):
        if isinstance(s, dict):
            meta_rows.append({
                "__idx__": i,
                "game_url": s.get("gameUrl") or s.get("url") or s.get("href"),
                "opponent": s.get("opponent") or s.get("opponentName") or s.get("name"),
                "home_away": s.get("homeAway") or s.get("location"),
                "game_date": s.get("gameDate") or s.get("date"),
            })
    meta = pd.DataFrame(meta_rows) if meta_rows else pd.DataFrame({"__idx__": body["__idx__"]})

    out = meta.merge(body, on="__idx__", how="outer").sort_values("__idx__")
    out["player_code"] = player_code
    out["game_no"] = out["__idx__"]
    out.drop(columns=["__idx__"], inplace=True, errors="ignore")

    # Ομογενοποίηση ονομάτων στηλών (ό,τι βρούμε το χαρτογραφούμε)
    alias_map = {
        "MIN":  ["MIN","Min","min","minutes","minutesPlayed"],
        "PTS":  ["PTS","Pts","pts","points","pointsScored"],
        "2FG":  ["2FG","2fg","twoPoints","twoPointers","twoPointersText","twoPointersMadeAttempted"],
        "3FG":  ["3FG","3fg","threePoints","threePointers","threePointersText","threePointersMadeAttempted"],
        "FT":   ["FT","ft","freeThrows","freeThrowsText","freeThrowsMadeAttempted"],
        "O":    ["O","o","offensive","offensiveRebounds"],
        "D":    ["D","d","defensive","defensiveRebounds"],
        "T":    ["T","t","total","totalRebounds"],
        "AST":  ["AST","Ast","ast","assists"],
        "STL":  ["STL","Stl","stl","steals"],
        "TOV":  ["TOV","To","TO","to","turnovers"],
        "BLK_FV":["Fv","fv","blocksFor","blocksMade","blocks"],
        "BLK_AG":["Ag","ag","blocksAgainst","blocksReceived"],
        "FLS_CM":["Cm","cm","foulsCommitted","foulsCommited"],
        "FLS_RV":["Rv","rv","foulsDrawn","foulsReceived"],
        "PIR":  ["PIR","pir","indexRating"],
    }
    rename = {}
    for std, keys in alias_map.items():
        for c in list(out.columns):
            if str(c) in keys:
                rename[c] = std; break
    out = out.rename(columns=rename)

    # Διαχωρισμός x/y για 2FG/3FG/FT
    def split_made_att(val):
        if isinstance(val, str) and "/" in val:
            a,b = val.split("/",1)
            try: return float(a), float(b)
            except: return a,b
        return None, None
    for cat in ("2FG","3FG","FT"):
        if cat in out.columns:
            m,a = zip(*[split_made_att(x) for x in out[cat].tolist()])
            out[f"{cat}_M"] = m; out[f"{cat}_A"] = a

    # σιγουριά ότι υπάρχουν όλα τα standards (έστω NA)
    for col in alias_map.keys():
        if col not in out.columns: out[col] = pd.NA

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-url", default="", help="CMS JSON URL (ένα παίκτη)")
    ap.add_argument("--json-file", default="", help="Τοπικό CMS JSON (π.χ. in/003458.json)")
    ap.add_argument("--player-code", default="", help="player_code για αναφορά")
    ap.add_argument("--season", default="2025")
    ap.add_argument("--competition", default="E")
    ap.add_argument("--mode", default="perGame")
    ap.add_argument("--out", default="out/")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.json_file:
        with open(args.json_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    elif args.json_url:
        r = requests.get(args.json_url, timeout=30)
        r.raise_for_status()
        payload = r.json()
    else:
        print("ERROR: δώσε --json-file ή --json-url")
        sys.exit(1)

    df = parse_player_gamelogs_from_content(payload, player_code=args.player_code or None)
    if df.empty:
        print("No gamelog rows parsed.")
        sys.exit(0)

    out_csv = os.path.join(args.out, f"player_gamelogs_{args.season}_{args.mode}.csv")
    # append αν υπάρχει ήδη
    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        df = pd.concat([old, df], ignore_index=True)

    df["season"] = args.season
    df["competition"] = args.competition
    df["mode"] = args.mode
    df.to_csv(out_csv, index=False)
    print(f"[ok] saved {out_csv} rows={len(df)}")

if __name__ == "__main__":
    main()
