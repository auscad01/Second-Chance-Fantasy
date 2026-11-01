"""Second Chance engine v2: data pull, features (3 optional groups), models, scoring.

Newest nflverse season is auto-detected, so new seasons and weeks flow through with no code changes.
Feature groups (see GROUPS / ACTIVE) were each tested walk-forward by research.py before being switched on.
"""
from __future__ import annotations
import datetime as dt, io, re, difflib, warnings
from pathlib import Path
import numpy as np, pandas as pd, requests
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

REL = "https://github.com/nflverse/nflverse-data/releases/download/"
URL = REL + "stats_player/stats_player_week_{y}.{ext}"
SNAP_URL = REL + "snap_counts/snap_counts_{y}.{ext}"
PLAYERS_URL = REL + "players/players.csv"
SCHED_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
CACHE = Path(__file__).parent / "data"
COLS = ["player_id", "player_display_name", "position", "season", "week", "season_type", "team", "opponent_team",
        "fantasy_points_ppr", "carries", "targets", "target_share", "air_yards_share", "wopr", "attempts",
        "passing_tds", "rushing_tds", "receiving_tds", "passing_interceptions", "sacks_suffered",
        "receiving_air_yards", "passing_air_yards", "rushing_yards", "receiving_yards", "receptions"]
NUM = COLS[8:]
POS = ["QB", "RB", "WR", "TE"]
TRAIN_FROM, MIN_PPG, MIN_G, UP, DOWN, FLAG_MULT, TEAMS_DONE = 2006, 6.0, 3, 1.2, 0.8, 1.6, 26
SNAPS_FROM = 2012
TEAMFIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
XFP_IN = ["carries", "targets", "receiving_air_yards", "attempts", "passing_air_yards"]

FEATS_BASE = ["pre_ppg", "g_pre", "pre_sd", "pre_max", "trend", "td_share", "opp_pg", "ppo", "carries_pg", "targets_pg",
              "tshare", "aysh", "wopr", "att_pg", "int_pg", "sack_pg", "prior_ppg", "prior_g", "vs_prior",
              "pre_minus_prior", "exp", "rookie", "is_QB", "is_RB", "is_WR", "is_TE"]
GROUPS = {
    "xfp": ["xfp_pg", "fpoe_pg", "fpoe_ratio", "xfp_trend", "prior_xfp_pg", "xfp_vs_prior"],
    "usage": ["tshare_trend", "carries_trend", "targets_trend", "snap_pg", "snap_last3", "snap_trend"],
    "team": ["team_pass_rate", "team_ppg", "team_ppg_vs_prior"],
    # --- candidates added from the Reddit "breakout stats" thread and the Kitman Labs draft-vs-success study ---
    "prior_late": ["prior_late_tshare", "prior_late_ppg", "prior_late_vs_early"],
    "airyds": ["adot", "air_yds_pg", "air_yds_gap", "racr"],
    "snapgap": ["snap_tgt_gap"],
    "competition": ["team_target_rank", "tgt_competition", "qb_quality"],
    "teammate_loss": ["lost_teammate_pts"],
    "rbypc": ["rb_ypc", "rb_ypc_gap", "rb_carry_rank"],
    "kitman": ["qb_rush_pg", "qb_rush_share", "rec_yds_pg", "yds_per_target", "rec_td_pg", "rb_rush_pts_share"],
}
# Round 1: tested walk-forward with research.py at cutoffs Week 4, 6 and 9 (2016-2025 folds, bootstrap CIs). Expected points
# helped at Week 6 only (+0.011 AUC on "beat expectation"); at Weeks 4 and 9, and for usage trends and team context
# everywhere, no group improved accuracy reliably.
# Round 2 (Nov 2026): prior_late, airyds, snapgap, competition, teammate_loss, rbypc and kitman, same cutoffs and rules.
# Only kitman cleared the bar, and only at Week 6 (+0.010 beat AUC, 90% lower bound +0.002); it was -0.004 at Week 4 and
# -0.006 at Week 9. A group must pass at two of three cutoffs to be switched on, so nothing is. Re-run research.py as seasons accrue.
ACTIVE = []


def feats():
    return FEATS_BASE + [f for g in ACTIVE for f in GROUPS[g]]


# ================================================================ data
def _get(url):
    try:
        r = requests.get(url, timeout=90)
        return r.content if r.status_code == 200 and len(r.content) > 1000 else None
    except Exception:
        return None


def _download(tmpl, y):
    for ext in ("parquet", "csv"):
        b = _get(tmpl.format(y=y, ext=ext))
        if b:
            return pd.read_parquet(io.BytesIO(b)) if ext == "parquet" else pd.read_csv(io.BytesIO(b), low_memory=False)
    return None


def _fresh(p, hours):
    return p.exists() and (hours is None or (dt.datetime.now().timestamp() - p.stat().st_mtime) < hours * 3600)


def get_season(y, refresh_hours=None):
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{y}.parquet"
    if _fresh(p, refresh_hours):
        c = pd.read_parquet(p)
        if "rushing_yards" in c.columns:
            return c
    df = _download(URL, y)
    if df is None:
        return pd.read_parquet(p) if p.exists() else None
    df = df[[c for c in COLS if c in df.columns]]
    df = df[(df.season_type == "REG") & df.position.isin(POS)].copy()
    df["team"] = df.team.replace(TEAMFIX)
    df["opponent_team"] = df.opponent_team.replace(TEAMFIX)
    df[NUM] = df[NUM].fillna(0)
    df["pts"] = df.fantasy_points_ppr
    df["tdpts"] = 6 * (df.rushing_tds + df.receiving_tds) + 4 * df.passing_tds
    df.to_parquet(p)
    return df


def latest_season():
    y0 = dt.date.today().year
    for y in (y0, y0 - 1, y0 - 2):
        df = get_season(y, refresh_hours=3)
        if df is not None and len(df) and df.week.max() >= 1:
            return y, df
    raise RuntimeError("Could not reach nflverse data.")


def completed_weeks(df):
    t = df.groupby("week").team.nunique()
    k = 0
    for w in sorted(w for w, n in t.items() if n >= TEAMS_DONE):
        if w == k + 1:
            k = w
    return k


def get_players_map():
    """pfr_id -> gsis_id, so snap counts (keyed by PFR) can join the weekly stats (keyed by GSIS)."""
    CACHE.mkdir(exist_ok=True)
    p = CACHE / "pfr_map.parquet"
    if _fresh(p, 24 * 7):
        return pd.read_parquet(p)
    b = _get(PLAYERS_URL)
    if b is None:
        return pd.read_parquet(p) if p.exists() else None
    m = pd.read_csv(io.BytesIO(b), usecols=["gsis_id", "pfr_id"], low_memory=False).dropna().drop_duplicates("pfr_id")
    m.to_parquet(p)
    return m


def get_snaps(y, pmap, refresh_hours=None):
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"snaps_{y}.parquet"
    if _fresh(p, refresh_hours):
        return pd.read_parquet(p)
    s = _download(SNAP_URL, y)
    if s is None or pmap is None:
        return pd.read_parquet(p) if p.exists() else None
    s = s[s.game_type == "REG"].merge(pmap, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    s = s.groupby(["gsis_id", "week"]).offense_pct.max().reset_index().rename(columns={"gsis_id": "player_id", "offense_pct": "snap_pct"})
    s.to_parquet(p)
    return s


def get_schedule(refresh_hours=6):
    CACHE.mkdir(exist_ok=True)
    p = CACHE / "schedule.parquet"
    if _fresh(p, refresh_hours):
        c = pd.read_parquet(p)
        if "pts_for" in c.columns:
            return c
    try:
        g = pd.read_csv(SCHED_URL, usecols=["season", "week", "game_type", "away_team", "home_team", "away_score", "home_score"])
    except Exception:
        return pd.read_parquet(p) if p.exists() else None
    g = g[g.game_type == "REG"].copy()
    for c in ("away_team", "home_team"):
        g[c] = g[c].replace(TEAMFIX)
    h = g.rename(columns={"home_team": "team", "away_team": "opp", "home_score": "pts_for", "away_score": "pts_against"})
    a = g.rename(columns={"away_team": "team", "home_team": "opp", "away_score": "pts_for", "home_score": "pts_against"})
    L = pd.concat([h, a])[["season", "week", "team", "opp", "pts_for", "pts_against"]]
    L.to_parquet(p)
    return L


# ================================================================ expected points (own model)
def _xfp_X(d):
    X = d[XFP_IN].copy()
    for p in POS:
        X["is_" + p] = (d.position == p).astype(int)
    return X


def fit_xfp(frames):
    """Points an average player scores from this much opportunity. Fit on 2006-2013 only so it never sees the test seasons."""
    tr = pd.concat([frames[y] for y in range(TRAIN_FROM, 2014) if y in frames])
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=.06, max_iter=200, min_samples_leaf=50, random_state=0).fit(_xfp_X(tr), tr.pts)


# ================================================================ features
def sos_table(d, sched, K):
    if sched is None or sched.empty:
        return None
    w = d[d.week <= K].groupby(["opponent_team", "position", "week"]).pts.sum().reset_index()
    a = w.groupby(["opponent_team", "position"]).pts.mean()
    lg = a.groupby("position").mean()
    r = a / a.index.get_level_values(1).map(lg).values
    r = 1 + (r - 1) * K / (K + 8)
    rem = sched[sched.week > K]
    out = []
    for pos in POS:
        rp = r.xs(pos, level=1)
        x = rem.assign(r=rem.opp.map(rp).fillna(1.0)).sort_values("week")
        g = x.groupby("team")
        out.append(pd.DataFrame({"sos_rem": g.r.mean(), "sos_next4": g.r.apply(lambda v: v.head(4).mean()),
                                 "rem_games": g.size()}).assign(pos=pos).reset_index())
    return pd.concat(out)


def team_table(d, sy, sp, K):
    pre = d[d.week <= K]
    t = pre.groupby("team").agg(att=("attempts", "sum"), car=("carries", "sum")).reset_index()
    t["team_pass_rate"] = t.att / (t.att + t.car).clip(lower=1)
    if sy is not None and "pts_for" in sy:
        t = t.merge(sy[sy.week <= K].groupby("team").pts_for.mean().rename("team_ppg").reset_index(), on="team", how="left")
        if sp is not None and len(sp):
            t = t.merge(sp.groupby("team").pts_for.mean().rename("_pp").reset_index(), on="team", how="left")
            t["team_ppg_vs_prior"] = t.team_ppg / t._pp
    return t[[c for c in ("team", "team_pass_rate", "team_ppg", "team_ppg_vs_prior") if c in t]]


def _trends(pre, cols):
    p = pre.sort_values(["player_id", "week"]).copy()
    p["_i"] = p.groupby("player_id").cumcount()
    p["_n"] = p.groupby("player_id").week.transform("count")
    f = p[p._i < 3].groupby("player_id")[cols].mean()
    l = p[p._i >= p._n - 3].groupby("player_id")[cols].mean()
    out = l - f
    n = p.groupby("player_id")._n.first().reindex(out.index)
    for c in cols:
        out.loc[n < 4, c] = np.nan
    return out, l


def _prior_table(d):
    """Last season's per-player summary, including how he finished (weeks 10+) vs how he started (weeks 1-9)."""
    base = d.groupby("player_id").agg(prior_g=("week", "count"), prior_ppg=("pts", "mean"), prior_xfp_pg=("xfp", "mean")).reset_index()
    L = d[d.week >= 10].groupby("player_id").agg(_lg=("week", "count"), prior_late_ppg=("pts", "mean"), prior_late_tshare=("target_share", "mean"))
    E = d[d.week <= 9].groupby("player_id").agg(_eg=("week", "count"), _early=("pts", "mean"))
    X = L.join(E, how="outer")
    X.loc[X._lg.fillna(0) < 3, ["prior_late_ppg", "prior_late_tshare"]] = np.nan
    X.loc[X._eg.fillna(0) < 3, "_early"] = np.nan
    X["prior_late_vs_early"] = (X.prior_late_ppg / X._early.clip(lower=1)).replace([np.inf, -np.inf], np.nan).clip(upper=5)
    return base.merge(X[["prior_late_ppg", "prior_late_tshare", "prior_late_vs_early"]].reset_index(), on="player_id", how="left")


PRIOR_COLS = ["player_id", "prior_g", "prior_ppg", "prior_xfp_pg", "prior_late_ppg", "prior_late_tshare", "prior_late_vs_early"]


def _extra(pre, A):
    """Candidate feature columns. Each group is tested by research.py; only groups listed in ACTIVE reach the model."""
    S = pre.groupby("player_id").agg(
        rush_yds=("rushing_yards", "sum"), rush_td=("rushing_tds", "sum"), car=("carries", "sum"),
        rec_yds=("receiving_yards", "sum"), tgt=("targets", "sum"), air=("receiving_air_yards", "sum"),
        rec_yds_pg=("receiving_yards", "mean"), rec_td_pg=("receiving_tds", "mean"),
        air_yds_pg=("receiving_air_yards", "mean"), rush_yds_pg=("rushing_yards", "mean"))
    A = A.merge(S, left_on="player_id", right_index=True, how="left").reset_index(drop=True)
    isq, isr, nq = A.pos == "QB", A.pos == "RB", A.pos != "QB"
    rp = 0.1 * A.rush_yds + 6 * A.rush_td
    # air yards: how deep his targets are, and whether production is lagging the opportunity (RACR < 1)
    A["adot"] = np.where(nq & (A.tgt >= 5), A.air / A.tgt.clip(lower=1), np.nan)
    A["racr"] = np.where(nq & (A.air >= 20), (A.rec_yds / A.air.clip(lower=1)).clip(0, 3), np.nan)
    A["air_yds_gap"] = np.where(nq, A.aysh - A.tshare, np.nan)
    A["air_yds_pg"] = np.where(nq, A.air_yds_pg, np.nan)
    # Kitman: QB rushing, receiving yardage, and RB rush-vs-catch mix
    A["qb_rush_pg"] = np.where(isq, A.rush_yds_pg, np.nan)
    A["qb_rush_share"] = np.where(isq, rp / A.pts.clip(lower=1), np.nan)
    A["rb_rush_pts_share"] = np.where(isr, rp / A.pts.clip(lower=1), np.nan)
    A["yds_per_target"] = np.where(nq & (A.tgt >= 5), A.rec_yds / A.tgt.clip(lower=1), np.nan)
    A["rec_yds_pg"] = np.where(nq, A.rec_yds_pg, np.nan)
    A["rec_td_pg"] = np.where(nq, A.rec_td_pg, np.nan)
    # RB yards per carry (shrunk toward the league mean) vs the team's lead back, and carry rank
    lg = A.loc[isr, "rush_yds"].sum() / max(A.loc[isr, "car"].sum(), 1)
    A["rb_ypc"] = np.where(isr, (A.rush_yds + 30 * lg) / (A.car + 30), np.nan)
    R = A.loc[isr, ["player_id", "team", "car", "rb_ypc"]].copy()
    R["rb_carry_rank"] = R.groupby("team").car.rank(ascending=False, method="first")
    lead = R[R.rb_carry_rank == 1].set_index("team").rb_ypc
    A = A.merge(R[["player_id", "rb_carry_rank"]], on="player_id", how="left").reset_index(drop=True)
    A["rb_ypc_gap"] = np.where(isr, A.rb_ypc - A.team.map(lead), np.nan)
    # competition for targets, and the quality of the QB throwing them
    A["team_target_rank"] = A.loc[nq].groupby("team").tgt.rank(ascending=False, method="first")
    pool = A[nq & (A.g_pre >= 2)][["player_id", "team", "tshare"]]
    comp = {}
    for t, g in pool.groupby("team"):
        v = g.tshare.fillna(0).values
        for i, pid in enumerate(g.player_id.values):
            comp[(t, pid)] = np.delete(v, i).max() if len(v) > 1 else np.nan
    A["tgt_competition"] = [comp.get((t, p), np.nan) if n else np.nan for t, p, n in zip(A.team, A.player_id, nq)]
    A["_att"] = A.att_pg * A.g_pre
    q = A[isq].sort_values("_att", ascending=False).drop_duplicates("team").set_index("team").prior_ppg
    A["qb_quality"] = np.where(nq, A.team.map(q), np.nan)
    # teammates at the same position who have missed the team's last 2 games (injury / benching) after playing 2+
    wk = pre[["team", "week"]].drop_duplicates().sort_values(["team", "week"])
    last2 = wk.groupby("team").tail(2).assign(_rec=1)
    P = pre[["player_id", "team", "week", "pts", "position"]].merge(last2, on=["team", "week"], how="left")
    P["_rec"] = P._rec.fillna(0)
    Q = P.groupby(["player_id", "team"]).agg(pos=("position", "last"), g=("week", "count"), rec=("_rec", "sum"), ppg=("pts", "mean")).reset_index()
    Q["cur"] = Q.player_id.map(A.set_index("player_id").team)
    out = Q[(Q.rec == 0) & (Q.g >= 2) & (Q.ppg >= 6) & (Q.cur == Q.team) & (Q.team.map(wk.groupby("team").size()) >= 4)]
    L = out.groupby(["team", "pos"]).ppg.sum().rename("_lost").reset_index()
    A = A.merge(L, on=["team", "pos"], how="left").reset_index(drop=True)
    A["lost_teammate_pts"] = A._lost.fillna(0) - A.player_id.map(out.set_index("player_id").ppg).fillna(0)
    return A.drop(columns=["_lost", "_att", "rush_yds", "rush_td", "car", "rec_yds", "tgt", "air", "rush_yds_pg"])


def features(d, prior, first, K, sos=None, team=None):
    pre = d[d.week <= K]
    A = pre.groupby("player_id").agg(
        name=("player_display_name", "last"), pos=("position", "last"), team=("team", "last"), g_pre=("week", "count"),
        pre_ppg=("pts", "mean"), pre_sd=("pts", "std"), pre_max=("pts", "max"), carries_pg=("carries", "mean"),
        targets_pg=("targets", "mean"), tshare=("target_share", "mean"), aysh=("air_yards_share", "mean"),
        wopr=("wopr", "mean"), att_pg=("attempts", "mean"), tdpts=("tdpts", "sum"), pts=("pts", "sum"),
        int_pg=("passing_interceptions", "mean"), sack_pg=("sacks_suffered", "mean"),
        xfp_pg=("xfp", "mean"), snap_pg=("snap_pct", "mean")).reset_index()
    tr, last = _trends(pre, ["pts", "xfp", "target_share", "carries", "targets", "snap_pct"])
    for col, new in (("pts", "trend"), ("xfp", "xfp_trend"), ("target_share", "tshare_trend"), ("carries", "carries_trend"),
                     ("targets", "targets_trend"), ("snap_pct", "snap_trend")):
        A[new] = A.player_id.map(tr[col])
    A["snap_last3"] = A.player_id.map(last["snap_pct"])
    A["td_share"] = A.tdpts / A.pts.clip(lower=1)
    A["opp_pg"] = A.carries_pg + A.targets_pg + 0.25 * A.att_pg
    A["ppo"] = A.pre_ppg / A.opp_pg.clip(lower=0.5)
    A["fpoe_pg"] = A.pre_ppg - A.xfp_pg
    A["fpoe_ratio"] = A.pre_ppg / A.xfp_pg.clip(lower=0.5)
    A = A.merge(prior, on="player_id", how="left")
    if sos is not None:
        A = A.merge(sos, left_on=["team", "pos"], right_on=["team", "pos"], how="left")
    if team is not None:
        A = A.merge(team, on="team", how="left")
    for c in ("sos_rem", "sos_next4", "rem_games", "team_pass_rate", "team_ppg", "team_ppg_vs_prior"):
        if c not in A:
            A[c] = np.nan
    season = int(d.season.iloc[0])
    A["exp"] = (season - A.player_id.map(first)).clip(0, 12)
    A["vs_prior"] = (A.pre_ppg / A.prior_ppg).replace([np.inf, -np.inf], np.nan)
    A["xfp_vs_prior"] = (A.xfp_pg / A.prior_xfp_pg).replace([np.inf, -np.inf], np.nan)
    A["pre_minus_prior"] = A.pre_ppg - A.prior_ppg
    A["rookie"] = A.prior_ppg.isna().astype(int)
    for p in POS:
        A["is_" + p] = (A.pos == p).astype(int)
    A = _extra(pre, A)
    A = A[(A.g_pre >= MIN_G) & (A.pre_ppg >= MIN_PPG)].copy()
    # snap share vs target share, ranked within position among scored players (WR/TE/RB)
    pr = lambda c: A.groupby("pos")[c].rank(pct=True)
    A["snap_tgt_gap"] = np.where(A.pos == "QB", np.nan, pr("snap_pg") - pr("tshare"))
    return A


# ================================================================ context + training table
def prepare(everything=False):
    """everything=True also loads snap counts (only needed to test the usage group)."""
    cur, dcur = latest_season()
    K = completed_weeks(dcur)
    frames = {cur: dcur}
    for y in range(TRAIN_FROM, cur):
        f = get_season(y)
        if f is not None:
            frames[y] = f
    use_snaps = everything or "usage" in ACTIVE or "snapgap" in ACTIVE
    pmap = get_players_map() if use_snaps else None
    for y in list(frames):
        d = frames[y]
        s = get_snaps(y, pmap, 3 if y == cur else None) if (use_snaps and y >= SNAPS_FROM) else None
        frames[y] = d.merge(s, on=["player_id", "week"], how="left") if s is not None else d.assign(snap_pct=np.nan)
    xm = fit_xfp(frames)
    for y, d in frames.items():
        d["xfp"] = xm.predict(_xfp_X(d))
    first = pd.concat(f[["player_id", "season"]] for f in frames.values()).groupby("player_id").season.min()
    return dict(cur=cur, K=K, frames=frames, first=first, sched=get_schedule(), maxwk=int(dcur.week.max()))


def season_features(ctx, y, Kc):
    f, sch = ctx["frames"], ctx["sched"]
    d = f[y]
    sy = None if sch is None else sch[sch.season == y]
    sp = None if sch is None else sch[sch.season == y - 1]
    prior = _prior_table(f[y - 1]) if (y - 1) in f else pd.DataFrame(columns=PRIOR_COLS)
    return features(d, prior, ctx["first"], Kc, sos_table(d, sy, Kc), team_table(d, sy, sp, Kc))


def build_training(ctx, Kt):
    rows = []
    for y in sorted(ctx["frames"]):
        if y == ctx["cur"] or (y - 1) not in ctx["frames"]:
            continue
        d = ctx["frames"][y]
        A = season_features(ctx, y, Kt)
        post = d[d.week > Kt].groupby("player_id").agg(g_post=("week", "count"), post_ppg=("pts", "mean")).reset_index()
        A = A.merge(post, on="player_id", how="left")
        need = max(2, int(0.4 * (int(d.week.max()) - Kt)))
        A = A[A.g_post >= need].copy()
        A["ratio"] = A.post_ppg / A.pre_ppg
        A["season"] = y
        rows.append(A)
    T = pd.concat(rows, ignore_index=True)
    T["heat"] = (T.ratio >= UP).astype(int)
    T["cool"] = (T.ratio <= DOWN).astype(int)
    return T


# ================================================================ models
BEATX = ["pre_ppg", "prior_f", "rookie", "is_QB", "is_RB", "is_WR"]


def _bx(df):
    x = df.copy()
    x["prior_f"] = x.prior_ppg.fillna(x.pre_ppg)
    return x[BEATX]


def add_beat(tr, te=None):
    """'Beat expectation': rest-of-season PPG 20%+ above what early PPG and last year's PPG alone predict (a plain
    linear baseline). This removes the free lunch from a slow start, so only real signal can score on it."""
    lr = LinearRegression().fit(_bx(tr), tr.post_ppg)
    f = lambda d: (d.post_ppg / lr.predict(_bx(d)).clip(min=1) >= UP).astype(int)
    out = [tr.assign(beat=f(tr))]
    if te is not None:
        out.append(te.assign(beat=f(te)))
    return out, lr


def _mk():
    return [HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=150, min_samples_leaf=40,
                                           l2_regularization=1.0, random_state=0),
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=.1, max_iter=500))]


def fit_predict(tr, te, target, cols=None):
    cols = cols or feats()
    return np.mean([m.fit(tr[cols], tr[target]).predict_proba(te[cols])[:, 1] for m in _mk()], axis=0)


def walk_forward(T, cols=None, n_folds=10):
    """Pooled out-of-sample predictions: each season predicted using only earlier seasons."""
    ys = sorted(T.season.unique())
    folds = [y for y in ys if y >= ys[-1] - (n_folds - 1) and (T.season < y).sum() > 800]
    out = {t: dict(y=[], p=[], s=[]) for t in ("heat", "cool", "beat")}
    for y in folds:
        (tr, te), _ = add_beat(T[T.season < y], T[T.season == y])
        for t in out:
            out[t]["y"] += list(te[t]); out[t]["p"] += list(fit_predict(tr, te, t, cols)); out[t]["s"] += [y] * len(te)
    return {t: {k: np.array(v) for k, v in d.items()} for t, d in out.items()}, folds


def _why(r, K):
    out = []
    if pd.notna(r.vs_prior):
        if r.vs_prior < .75:
            out.append(f"{r.pre_ppg:.1f} PPG vs {r.prior_ppg:.1f} last year")
        elif r.vs_prior > 1.3:
            out.append(f"{r.vs_prior - 1:.0%} above last year's pace")
    if r.td_share >= .4:
        out.append(f"{r.td_share:.0%} of points from TDs")
    if pd.notna(r.sos_rem) and r.sos_rem >= 1.06:
        out.append(f"soft schedule ahead (+{r.sos_rem - 1:.0%})")
    elif pd.notna(r.sos_rem) and r.sos_rem <= 0.94:
        out.append(f"tough schedule ahead ({r.sos_rem - 1:.0%})")
    if r.g_pre < K - 1:
        out.append(f"played {int(r.g_pre)} of {K} weeks")
    return "; ".join(out)


def run():
    ctx = prepare()
    cur, K = ctx["cur"], ctx["K"]
    meta = dict(season=cur, K=K, max_week=ctx["maxwk"], asof=dt.datetime.now(), seasons=len(ctx["frames"]) - 1, groups=list(ACTIVE))
    dcur = ctx["frames"][cur]
    allp = dcur.groupby("player_id").agg(name=("player_display_name", "last"), pos=("position", "last"), team=("team", "last")).reset_index()
    if K < 3 or K >= 17:
        meta["status"] = "early" if K < 3 else "complete"
        return dict(meta=meta, scored=pd.DataFrame(), allp=allp, val={}, base={})
    T = build_training(ctx, min(K, 14))
    wf, folds = walk_forward(T)
    val = {}
    for t, d in wf.items():
        thr = np.quantile(d["p"], .9)
        val[t] = dict(auc=roc_auc_score(d["y"], d["p"]), top=d["y"][d["p"] >= thr].mean(), base=d["y"].mean(), n=len(d["y"]), folds=f"{folds[0]}-{folds[-1]}")
    (Tb,), _ = add_beat(T)
    base = dict(heat=T.heat.mean(), cool=T.cool.mean(), beat=Tb.beat.mean(), n=len(T))
    S = season_features(ctx, cur, K)
    S["p_heat"], S["p_cool"], S["p_beat"] = (fit_predict(Tb, S, t) for t in ("heat", "cool", "beat"))
    S["flag"] = np.where((S.p_heat >= FLAG_MULT * base["heat"]) & (S.p_cool >= FLAG_MULT * base["cool"]), "VOLATILE",
                np.where(S.p_heat >= FLAG_MULT * base["heat"], "HEAT UP",
                np.where(S.p_cool >= FLAG_MULT * base["cool"], "COOL DOWN", "STEADY")))
    S["why"] = S.apply(lambda r: _why(r, K), axis=1)
    meta["status"] = "ok"
    return dict(meta=meta, scored=S.sort_values("p_heat", ascending=False).reset_index(drop=True), allp=allp, val=val, base=base)


# ================================================================ roster import (Streamlit version)
def _norm(s):
    s = re.sub(r"[^a-z ]", "", str(s).lower().replace("-", " "))
    return " ".join(re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s).split())


def match_names(names, allp):
    keys = {}
    for r in allp.itertuples():
        keys.setdefault(_norm(r.name), r.player_id)
    hit, miss = [], []
    for n in names:
        k = _norm(n)
        if not k:
            continue
        if k in keys:
            hit.append(keys[k])
        else:
            c = difflib.get_close_matches(k, list(keys), n=1, cutoff=.84)
            (hit if c else miss).append(keys[c[0]] if c else n)
    return list(dict.fromkeys(hit)), miss
