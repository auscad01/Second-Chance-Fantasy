"""Second Chance engine: data pull, features, models, scoring.

Everything keys off the *latest season nflverse has published*, so new seasons
(and new weeks) flow through automatically with no code changes.
"""
from __future__ import annotations
import datetime as dt, io, re, difflib, warnings
from pathlib import Path
import numpy as np, pandas as pd, requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{y}.{ext}"
CACHE = Path(__file__).parent / "data"
COLS = ["player_id", "player_display_name", "position", "season", "week", "season_type", "team",
        "opponent_team", "fantasy_points_ppr", "carries", "targets", "target_share", "air_yards_share", "wopr", "attempts",
        "passing_tds", "rushing_tds", "receiving_tds", "passing_interceptions", "sacks_suffered"]
POS = ["QB", "RB", "WR", "TE"]
TRAIN_FROM, MIN_PPG, MIN_G, UP, DOWN, TEAMS_DONE, FLAG_MULT = 2006, 6.0, 3, 1.2, 0.8, 26, 1.6
FEATS = ["pre_ppg", "g_pre", "pre_sd", "pre_max", "trend", "td_share", "opp_pg", "ppo", "carries_pg", "targets_pg",
         "tshare", "aysh", "wopr", "att_pg", "int_pg", "sack_pg", "prior_ppg", "prior_g", "vs_prior",
         "pre_minus_prior", "exp", "rookie", "is_QB", "is_RB", "is_WR", "is_TE"] 
# Schedule softness (sos_*) is computed for display only: walk-forward tests showed it adds no accuracy.
TEAMFIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
SCHED_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


# ---------------------------------------------------------------- data
def _download(y):
    for ext in ("parquet", "csv"):
        try:
            r = requests.get(URL.format(y=y, ext=ext), timeout=90)
            if r.status_code == 200 and len(r.content) > 1000:
                b = io.BytesIO(r.content)
                return pd.read_parquet(b) if ext == "parquet" else pd.read_csv(b, low_memory=False)
        except Exception:
            continue
    return None


def get_season(y, refresh_hours=None):
    """Weekly stats for one season. Past seasons are cached forever; the live one refreshes."""
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{y}.parquet"
    fresh = p.exists() and (refresh_hours is None or (dt.datetime.now().timestamp() - p.stat().st_mtime) < refresh_hours * 3600)
    if fresh:
        c = pd.read_parquet(p)
        if "opponent_team" in c.columns:
            return c
    df = _download(y)
    if df is None:
        return pd.read_parquet(p) if p.exists() else None
    df = df[[c for c in COLS if c in df.columns]]
    df = df[(df.season_type == "REG") & df.position.isin(POS)].copy()
    df["team"] = df.team.replace(TEAMFIX)
    df["opponent_team"] = df.opponent_team.replace(TEAMFIX)
    num = [c for c in COLS[7:] if c in df.columns]
    df[num] = df[num].fillna(0)
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
    done = [w for w, n in t.items() if n >= TEAMS_DONE]
    k = 0
    for w in sorted(done):
        if w == k + 1:
            k = w
    return k


# ---------------------------------------------------------------- schedule
def get_schedule(refresh_hours=6):
    """Regular-season schedule in long form: one row per team-game (season, week, team, opp)."""
    CACHE.mkdir(exist_ok=True)
    p = CACHE / "schedule.parquet"
    if p.exists() and (dt.datetime.now().timestamp() - p.stat().st_mtime) < refresh_hours * 3600:
        return pd.read_parquet(p)
    try:
        g = pd.read_csv(SCHED_URL, usecols=["season", "week", "game_type", "away_team", "home_team"])
    except Exception:
        return pd.read_parquet(p) if p.exists() else None
    g = g[g.game_type == "REG"]
    for c in ("away_team", "home_team"):
        g[c] = g[c].replace(TEAMFIX)
    L = pd.concat([g.rename(columns={"home_team": "team", "away_team": "opp"})[["season", "week", "team", "opp"]],
                   g.rename(columns={"away_team": "team", "home_team": "opp"})[["season", "week", "team", "opp"]]])
    L.to_parquet(p)
    return L


def sos_table(d, sched, K):
    """Remaining-schedule softness by (team, pos). 1.00 = average; above 1 = opponents have allowed more points to that position."""
    if sched is None or sched.empty:
        return None
    w = d[d.week <= K].groupby(["opponent_team", "position", "week"]).pts.sum().reset_index()
    a = w.groupby(["opponent_team", "position"]).pts.mean()
    lg = a.groupby("position").mean()
    r = a / a.index.get_level_values(1).map(lg).values
    r = 1 + (r - 1) * K / (K + 8)  # early-season defensive stats are noisy: shrink toward average
    rem = sched[sched.week > K]
    out = []
    for pos in POS:
        rp = r.xs(pos, level=1)
        x = rem.assign(r=rem.opp.map(rp).fillna(1.0)).sort_values("week")
        g = x.groupby("team")
        out.append(pd.DataFrame({"sos_rem": g.r.mean(), "sos_next4": g.r.apply(lambda v: v.head(4).mean()),
                                 "rem_games": g.size()}).assign(pos=pos).reset_index())
    return pd.concat(out)


# ---------------------------------------------------------------- features
def _last3(g):
    g = g.sort_values("week")
    return g.pts.tail(3).mean() - g.pts.head(3).mean() if len(g) >= 4 else np.nan


def features(d, prior, first, K, sos=None):
    pre = d[d.week <= K]
    A = pre.groupby("player_id").agg(
        name=("player_display_name", "last"), pos=("position", "last"), team=("team", "last"), g_pre=("week", "count"),
        pre_ppg=("pts", "mean"), pre_sd=("pts", "std"), pre_max=("pts", "max"), carries_pg=("carries", "mean"),
        targets_pg=("targets", "mean"), tshare=("target_share", "mean"), aysh=("air_yards_share", "mean"),
        wopr=("wopr", "mean"), att_pg=("attempts", "mean"), tdpts=("tdpts", "sum"), pts=("pts", "sum"),
        int_pg=("passing_interceptions", "mean"), sack_pg=("sacks_suffered", "mean")).reset_index()
    A["trend"] = A.player_id.map(pre.groupby("player_id").apply(_last3))
    A["td_share"] = A.tdpts / A.pts.clip(lower=1)
    A["opp_pg"] = A.carries_pg + A.targets_pg + 0.25 * A.att_pg
    A["ppo"] = A.pre_ppg / A.opp_pg.clip(lower=0.5)
    A = A.merge(prior, on="player_id", how="left")
    if sos is not None:
        A = A.merge(sos, on=["team", "pos"], how="left")
    for c in ("sos_rem", "sos_next4", "rem_games"):
        if c not in A:
            A[c] = np.nan
    season = int(d.season.iloc[0])
    A["exp"] = (season - A.player_id.map(first)).clip(0, 12)
    A["vs_prior"] = (A.pre_ppg / A.prior_ppg).replace([np.inf, -np.inf], np.nan)
    A["pre_minus_prior"] = A.pre_ppg - A.prior_ppg
    A["rookie"] = A.prior_ppg.isna().astype(int)
    for p in POS:
        A["is_" + p] = (A.pos == p).astype(int)
    return A[(A.g_pre >= MIN_G) & (A.pre_ppg >= MIN_PPG)].copy()


def _prior_table(d):
    g = d.groupby("player_id").agg(prior_g=("week", "count"), prior_ppg=("pts", "mean")).reset_index()
    return g


def _mk():
    return [HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=150, min_samples_leaf=40,
                                           l2_regularization=1.0, random_state=0),
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=.1, max_iter=500))]


def _fit_predict(tr, te, target):
    return np.mean([m.fit(tr[FEATS], tr[target]).predict_proba(te[FEATS])[:, 1] for m in _mk()], axis=0)


# ---------------------------------------------------------------- pipeline
def run():
    cur, dcur = latest_season()
    K = completed_weeks(dcur)
    frames = {cur: dcur}
    for y in range(TRAIN_FROM, cur):
        f = get_season(y)
        if f is not None:
            frames[y] = f
    first = pd.concat(frames.values()).groupby("player_id").season.min()
    sched = get_schedule()
    maxwk = int(dcur.week.max())
    meta = dict(season=cur, K=K, max_week=maxwk, asof=dt.datetime.now(), seasons=len(frames) - 1)
    allp = dcur.groupby("player_id").agg(name=("player_display_name", "last"), pos=("position", "last"), team=("team", "last")).reset_index()
    if K < 3 or K >= 17:
        meta["status"] = "early" if K < 3 else "complete"
        return dict(meta=meta, scored=pd.DataFrame(), allp=allp, val={}, base={})
    Kt = min(K, 14)
    rows = []
    for y in sorted(frames):
        if y == cur or (y - 1) not in frames:
            continue
        d = frames[y]
        A = features(d, _prior_table(frames[y - 1]), first, Kt, sos_table(d, None if sched is None else sched[sched.season == y], Kt))
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
    base = dict(heat=T.heat.mean(), cool=T.cool.mean(), n=len(T))
    # walk-forward validation on the last 6 seasons
    val, ys = {}, sorted(T.season.unique())
    folds = [y for y in ys if y >= ys[-1] - 5 and (T.season < y).sum() > 800]
    for tgt in ("heat", "cool"):
        yt, pp = [], []
        for y in folds:
            tr, te = T[T.season < y], T[T.season == y]
            yt += list(te[tgt]); pp += list(_fit_predict(tr, te, tgt))
        yt, pp = np.array(yt), np.array(pp)
        thr = np.quantile(pp, .9)
        val[tgt] = dict(auc=roc_auc_score(yt, pp), top=yt[pp >= thr].mean(), base=yt.mean(), n=len(yt), folds=f"{folds[0]}-{folds[-1]}")
    # score the live season
    prior = _prior_table(frames[cur - 1]) if (cur - 1) in frames else pd.DataFrame(columns=["player_id", "prior_g", "prior_ppg"])
    S = features(dcur, prior, first, K, sos_table(dcur, None if sched is None else sched[sched.season == cur], K))
    S["p_heat"] = _fit_predict(T, S, "heat")
    S["p_cool"] = _fit_predict(T, S, "cool")
    S["flag"] = np.where((S.p_heat >= FLAG_MULT * base["heat"]) & (S.p_cool >= FLAG_MULT * base["cool"]), "VOLATILE",
                np.where(S.p_heat >= FLAG_MULT * base["heat"], "HEAT UP",
                np.where(S.p_cool >= FLAG_MULT * base["cool"], "COOL DOWN", "STEADY")))
    S["why"] = S.apply(lambda r: _why(r, K), axis=1)
    meta["status"] = "ok"
    return dict(meta=meta, scored=S.sort_values("p_heat", ascending=False).reset_index(drop=True), allp=allp, val=val, base=base)


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


# ---------------------------------------------------------------- roster import
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
