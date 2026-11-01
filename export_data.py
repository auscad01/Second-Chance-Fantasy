"""Runs the Second Chance model and writes docs/data.json for the web app.
GitHub Actions runs this every 6 hours; run it by hand with `python export_data.py`."""
import json, sys
from pathlib import Path
import engine

OUT = Path(__file__).parent / "data.json"


def main():
    B = engine.run()
    M, S, A = B["meta"], B["scored"], B["allp"]
    cols = ["player_id", "name", "pos", "team", "g_pre", "pre_ppg", "prior_ppg", "p_heat", "p_cool", "flag", "sos_rem", "why"]
    players = json.loads(S[cols].round(3).to_json(orient="records")) if len(S) else []
    out = dict(
        meta=dict(season=int(M["season"]), K=int(M["K"]), status=M["status"], seasons=int(M["seasons"]), asof=M["asof"].isoformat(timespec="minutes")),
        base={k: float(v) for k, v in B["base"].items()},
        val={t: {k: (float(v) if not isinstance(v, str) else v) for k, v in d.items()} for t, d in B["val"].items()},
        flagx=engine.FLAG_MULT,
        players=players,
        all=json.loads(A[["player_id", "name", "pos", "team"]].to_json(orient="records")),
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB): {len(players)} scored, season {out['meta']['season']} wk {out['meta']['K']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # fail loudly so the Action doesn't commit stale/partial data
        print("export failed:", e)
        sys.exit(1)
