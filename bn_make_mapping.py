#!/usr/bin/env python3
import argparse, os, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True)
    ap.add_argument("--mode", default="perGame")
    ap.add_argument("--players_csv", default="")
    ap.add_argument("--out_csv", default="bn_map.csv")
    args = ap.parse_args()

    players_csv = args.players_csv or os.path.join("out", f"players_{args.season}_{args.mode}.csv")
    if not os.path.exists(players_csv):
        raise SystemExit(f"Δεν βρέθηκε {players_csv}. Τρέξε πρώτα το season step.")

    df = pd.read_csv(players_csv)
    keep = [c for c in df.columns if c in ("player_code","player_name","player_team_name","player_team_code") or "name" in c or "team" in c]
    out = df[keep].copy()
    if "player_code" not in out.columns:
        raise SystemExit("Δεν βρήκα στήλη player_code στο players CSV.")
    out["bn_path"] = ""  # π.χ. 19540-will-clyburn
    out.drop_duplicates(subset=["player_code"], inplace=True)
    out.sort_values("player_name" if "player_name" in out.columns else "player_code", inplace=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] έγραψα {args.out_csv} — συμπλήρωσε τη στήλη bn_path.")

if __name__ == "__main__":
    main()
