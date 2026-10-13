import streamlit as st
import pandas as pd
import numpy as np
import requests
import pulp

st.set_page_config(page_title="Project Top 10k", page_icon="⚽", layout="wide")
HEADERS = {"User-Agent": "Mozilla/5.0"}

@st.cache_data(ttl=1800)
def get_fpl_data():
    b = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=HEADERS, timeout=15).json()
    f = requests.get("https://fantasy.premierleague.com/api/fixtures/", headers=HEADERS, timeout=15).json()
    return b, f

def get_manager(mid, gw):
    base = f"https://fantasy.premierleague.com/api/entry/{mid}"
    r = requests.get(f"{base}/", headers=HEADERS, timeout=10)
    if r.status_code != 200:
        return None
    h = requests.get(f"{base}/history/", headers=HEADERS, timeout=10).json()
    p = requests.get(f"{base}/event/{max(1, gw)}/picks/", headers=HEADERS, timeout=10).json()
    return {"info": r.json(), "hist": h, "picks": p}

def calc_fts(hist):
    curr = hist.get("current", [])
    chips = {c["event"]: c["name"] for c in hist.get("chips", [])}
    ft = 1
    for i, row in enumerate(curr):
        if i == 0:
            ft = 1
        elif chips.get(row["event"]) in ["wildcard", "freehit"]:
            ft = min(5, max(1, ft))
        else:
            ft = min(5, max(0, ft - row.get("event_transfers", 0)) + 1)
    return int(np.clip(ft, 1, 5))

def build_model(boot, fixtures, next_gw, fix_w=0.45, mode="Balanced"):
    t_df = pd.DataFrame(boot["teams"])
    p_df = pd.DataFrame(boot["elements"])
    f_df = pd.DataFrame(fixtures)

    t_short = dict(zip(t_df["id"], t_df["short_name"]))
    att_h = dict(zip(t_df["id"], t_df["strength_attack_home"]))
    att_a = dict(zip(t_df["id"], t_df["strength_attack_away"]))
    def_h = dict(zip(t_df["id"], t_df["strength_defence_home"]))
    def_a = dict(zip(t_df["id"], t_df["strength_defence_away"]))
    avg_s = np.mean(list(att_h.values()) + list(def_h.values()))

    cols = [
        "now_cost", "selected_by_percent", "minutes", "form", "points_per_game",
        "expected_goals_per_90", "expected_assists_per_90",
        "expected_goal_involvements_per_90", "expected_goals_conceded_per_90",
        "saves_per_90", "bps", "penalties_order", "corners_and_indirect_freekicks_order"
    ]
    for c in cols:
        p_df[c] = pd.to_numeric(p_df.get(c, 0), errors="coerce").fillna(0.0)

    p_df["price"] = p_df["now_cost"] / 10.0
    p_df["pos"] = p_df["element_type"].map({1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"})
    p_df["club"] = p_df["team"].map(t_short)

    games = max(1, next_gw - 1)
    m_ratio = np.clip(p_df["minutes"] / (games * 90.0), 0.0, 1.0)
    chance = pd.to_numeric(p_df.get("chance_of_playing_next_round", 100), errors="coerce").fillna(100.0) / 100.0
    stat_m = p_df["status"].map({"a": 1.0, "d": 0.75}).fillna(0.0)
    p_df["mins_sec"] = np.clip((0.65 * m_ratio + 0.35 * np.minimum(1.0, p_df["minutes"] / 180.0)) * chance * stat_m, 0.0, 1.0)

    nineties = np.maximum(1.0, p_df["minutes"] / 90.0)
    if "defensive_contribution_per_90" in p_df.columns:
        dc = pd.to_numeric(p_df["defensive_contribution_per_90"], errors="coerce").fillna(0.0)
    else:
        dc = (p_df["bps"] / nineties) * 0.38
    p_df["defcon_90"] = dc.round(2)
    p_df["defcon_pts"] = np.where(p_df["pos"] == "DEF", np.clip(dc / 13.5, 0, 0.75) * 2.0,
                         np.where(p_df["pos"].isin(["MID", "FWD"]), np.clip(dc / 16.5, 0, 0.65) * 2.0, 0.0))

    sp_b = np.where(p_df["penalties_order"] == 1, 0.45, 0.0) + np.where(p_df["corners_and_indirect_freekicks_order"] == 1, 0.20, 0.0)
    g_pts = p_df["pos"].map({"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4})
    cs_pts = p_df["pos"].map({"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0})

    att_xp = p_df["expected_goals_per_90"] * g_pts + p_df["expected_assists_per_90"] * 3.0 + sp_b
    def_xp = np.clip(1.0 - (p_df["expected_goals_conceded_per_90"] / 2.2), 0.15, 0.60) * cs_pts + p_df["defcon_pts"]
    bonus_xp = np.clip(((p_df["bps"] / nineties) - 12.0) / 14.0, 0.0, 2.0)
    raw_xp = 2.0 + att_xp + def_xp + bonus_xp + np.where(p_df["pos"] == "GKP", p_df["saves_per_90"] / 3.0, 0.0)

    p_df["base_xp"] = np.clip(0.65 * raw_xp + 0.20 * p_df["points_per_game"] + 0.15 * p_df["form"], 0.5, 11.5) * p_df["mins_sec"]
    p_df["role"] = np.where(p_df["selected_by_percent"] >= 30, "🛡️ Shield", np.where(p_df["selected_by_percent"] <= 10, "⚔️ Sword", "⚖️ Core"))

    m_mult = np.ones(len(p_df))
    if "Sword" in mode:
        m_mult = np.where(p_df["selected_by_percent"] <= 10.0, 1.12, 0.96)
    elif "Shield" in mode:
        m_mult = np.where(p_df["selected_by_percent"] >= 25.0, 1.10, 0.95)

    gws = list(range(next_gw, min(39, next_gw + 5)))
    f_sub = f_df[f_df["event"].isin(gws)]
    t_fix = {t: {g: [] for g in gws} for t in t_df["id"]}

    for _, f in f_sub.iterrows():
        g, th, ta = int(f["event"]), int(f["team_h"]), int(f["team_a"])
        t_fix[th][g].append((f"{t_short[ta]}(H)", (avg_s / max(800, def_a[ta])) * 1.06, (avg_s / max(800, att_a[ta])) * 1.06))
        t_fix[ta][g].append((f"{t_short[th]}(A)", (avg_s / max(800, def_h[th])) * 0.95, (avg_s / max(800, att_h[th])) * 0.95))

    for g in gws:
        xp_l, op_l = [], []
        for idx, r in p_df.iterrows():
            fl = t_fix.get(int(r["team"]), {}).get(g, [])
            if not fl:
                xp_l.append(0.0)
                op_l.append("BLANK")
                continue
            tot, ops = 0.0, []
            for op, am, dm in fl:
                ops.append(op)
                fm = (0.7 * dm + 0.3 * am) if r["pos"] in ["GKP", "DEF"] else (0.85 * am + 0.15 * dm)
                tot += r["base_xp"] * (1.0 + fix_w * (fm - 1.0))
            xp_l.append(round(tot * m_mult[idx], 2))
            op_l.append("+".join(ops))
        p_df[f"xp_{g}"] = xp_l
        p_df[f"op_{g}"] = op_l

    p_df["xp_5gw"] = p_df[[f"xp_{g}" for g in gws]].sum(axis=1).round(2)
    p_df["fix_5gw"] = p_df[[f"op_{g}" for g in gws]].apply(lambda r: " → ".join(r.values), axis=1)
    q98 = max(1.0, p_df["xp_5gw"].quantile(0.98))
    p_df["score"] = np.clip((p_df["xp_5gw"] / q98) * 100.0, 0.0, 99.0).round(1)

    def get_verdict(r):
        if r["mins_sec"] < 0.65:
            return "🚨 SELL: Mins/Injury Risk"
        if r["xp_5gw"] < 16.5:
            return "🚨 SELL: Fixture Cliff / Low xP"
        if r["pos"] in ["MID", "FWD"] and r["form"] > 5.0 and r["expected_goal_involvements_per_90"] < 0.25:
            return "🚨 SELL: xGI Mirage"
        return "✅ STRONG HOLD" if r["score"] >= 72 else "⚖️ HOLD"

    p_df["verdict"] = p_df.apply(get_verdict, axis=1)
    return p_df, gws

def solve_xi(sq_df, gw_col):
    df = sq_df.copy().reset_index(drop=True)
    n = len(df)
    prob = pulp.LpProblem("XI", pulp.LpMaximize)
    s = [pulp.LpVariable(f"s_{i}", cat="Binary") for i in range(n)]
    c = [pulp.LpVariable(f"c_{i}", cat="Binary") for i in range(n)]
    xp = df[gw_col].values
    prob += pulp.lpSum([xp[i] * (s[i] + c[i]) for i in range(n)])
    prob += pulp.lpSum(s) == 11
    prob += pulp.lpSum(c) == 1
    for i in range(n):
        prob += c[i] <= s[i]
    for pos, lo, hi in [("GKP", 1, 1), ("DEF", 3, 5), ("MID", 2, 5), ("FWD", 1, 3)]:
        idx = [i for i in range(n) if df.loc[i, "pos"] == pos]
        prob += pulp.lpSum([s[i] for i in idx]) >= lo
        prob += pulp.lpSum([s[i] for i in idx]) <= hi
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    df["start"] = [int(pulp.value(s[i]) or 0) for i in range(n)]
    df["cap"] = [int(pulp.value(c[i]) or 0) for i in range(n)]
    s_xp = sum(xp[i] * (df.loc[i, "start"] + df.loc[i, "cap"]) for i in range(n))
    b_xp = sum(xp[i] * (1 - df.loc[i, "start"]) for i in range(n))
    return df, round(s_xp, 2), round(b_xp, 2)

def solve_transfers(all_df, curr_ids, bank, max_t, gws, gamma=0.90):
    c_set = set(curr_ids)
    top_ids = set(all_df.sort_values("xp_5gw", ascending=False).head(150)["id"])
    pool = all_df[all_df["id"].isin(c_set.union(top_ids))].copy().reset_index(drop=True)
    n = len(pool)
    disc = np.zeros(n)
    for i, g in enumerate(gws):
        disc += (gamma ** i) * pool[f"xp_{g}"].values
    pool["disc"] = disc

    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    s = [pulp.LpVariable(f"s_{i}", cat="Binary") for i in range(n)]
    c = [pulp.LpVariable(f"c_{i}", cat="Binary") for i in range(n)]
    prob = pulp.LpProblem("Transfers", pulp.LpMaximize)
    g1 = pool[f"xp_{gws[0]}"].values
    prob += pulp.lpSum([0.55 * g1[i] * (s[i] + c[i]) + 0.45 * pool.loc[i, "disc"] * (s[i] + 0.15 * x[i]) for i in range(n)])
    prob += pulp.lpSum(x) == 15
    prob += pulp.lpSum(s) == 11
    prob += pulp.lpSum(c) == 1
    for i in range(n):
        prob += s[i] <= x[i]
        prob += c[i] <= s[i]
    for pos, req, lo, hi in [("GKP", 2, 1, 1), ("DEF", 5, 3, 5), ("MID", 5, 2, 5), ("FWD", 3, 1, 3)]:
        idx = [i for i in range(n) if pool.loc[i, "pos"] == pos]
        prob += pulp.lpSum([x[i] for i in idx]) == req
        prob += pulp.lpSum([s[i] for i in idx]) >= lo
        prob += pulp.lpSum([s[i] for i in idx]) <= hi
    for tid in pool["team"].unique():
        idx = [i for i in range(n) if pool.loc[i, "team"] == tid]
        prob += pulp.lpSum([x[i] for i in idx]) <= 3

    bud = pool[pool["id"].isin(c_set)]["price"].sum() + bank
    prob += pulp.lpSum([pool.loc[i, "price"] * x[i] for i in range(n)]) <= bud + 0.001
    kept = [i for i in range(n) if int(pool.loc[i, "id"]) in c_set]
    prob += pulp.lpSum([x[i] for i in kept]) >= 15 - max_t
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    new_sq = pool.loc[[i for i in range(n) if (pulp.value(x[i]) or 0) > 0.5]].copy()
    n_ids = set(new_sq["id"])
    sold = pool[pool["id"].isin(c_set - n_ids)]
    bought = pool[pool["id"].isin(n_ids - c_set)]
    tot_xp = sum((gamma ** i) * solve_xi(new_sq, f"xp_{g}")[1] for i, g in enumerate(gws))
    return new_sq, sold, bought, round(tot_xp, 2)

st.title("⚽ Project Top 10k: FPL Command Centre")
boot, fixtures = get_fpl_data()
events = boot["events"]
next_ev = next((e for e in events if e.get("is_next")), events[-1])
next_gw = int(next_ev["id"])

with st.sidebar:
    st.header("Team & Model Settings")
    mid_in = st.text_input("FPL Manager ID", value="", placeholder="e.g. 1234567")
    mode = st.selectbox("Strategy Mode", ["Balanced", "Sword (<10% Diffs)", "Shield (High EO)"])
    fix_w = st.slider("Fixture Weighting", 0.1, 0.85, 0.45, 0.05)

p_df, gws = build_model(boot, fixtures, next_gw, fix_w, mode)

auto_ft, auto_bank, sq_ids = 1, 0.5, []
title = "Sample Squad (Tap '>' top-left to enter your Manager ID)"
if mid_in.strip().isdigit():
    md = get_manager(int(mid_in.strip()), next_gw - 1)
    if md and "picks" in md["picks"]:
        sq_ids = [p["element"] for p in md["picks"]["picks"]]
        auto_ft = calc_fts(md["hist"])
        auto_bank = md["picks"].get("entry_history", {}).get("bank", 0) / 10.0
        title = f"{md['info'].get('name', 'My Team')} (Banked FTs: {auto_ft})"

if not sq_ids:
    for pos, cnt in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
        sq_ids.extend(p_df[p_df["pos"] == pos].sort_values("selected_by_percent", ascending=False).head(cnt)["id"].tolist())

with st.sidebar:
    fts = st.slider("Free Transfers Available (1-5)", 1, 5, int(auto_ft))
    bank = st.number_input("Bank (£m)", 0.0, 25.0, float(auto_bank), 0.1)

curr_sq = p_df[p_df["id"].isin(sq_ids)].copy()
st.subheader(f"📋 {title}")

t1, t2, t3 = st.tabs([f"🚀 GW{next_gw} Roll & Chip Planner", "🪓 Chopping Block", "🔍 Target Hunter"])

with t1:
    _, _, _, xp_hold = solve_transfers(p_df, sq_ids, bank, 0, gws)
    sq_1, s_1, b_1, xp_1 = solve_transfers(p_df, sq_ids, bank, 1, gws)
    sq_m, s_m, b_m, xp_m = solve_transfers(p_df, sq_ids, bank, max(2, fts), gws)
    g1 = round(xp_1 - xp_hold, 2)
    gm = round(xp_m - xp_hold, 2)
    next_ft = min(5, fts + 1)

    if fts == 5:
        st.warning(f"🔥 **USE TRANSFER (At 5 FT Cap!)** Best 1-FT move gains **+{g1} xP** over 5 GWs.")
    elif g1 < 2.6:
        st.info(f"🏦 **VERDICT: ROLL TRANSFER ({fts} FT → {next_ft} FTs)** | 1 move only gains +{g1} xP, while a {max(2, fts)}-transfer swing gains **+{gm} xP**.")
    else:
        st.success(f"✅ **VERDICT: USE TRANSFER NOW (+{g1} xP Gain)**")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Roll (Bank {next_ft} FT)", f"{xp_hold} xP", "0 Transfers")
    o1 = ", ".join(s_1["web_name"]) if not s_1.empty else "Hold"
    i1 = ", ".join(b_1["web_name"]) if not b_1.empty else "Hold"
    c2.metric(f"1 FT: {o1} → {i1}", f"{xp_1} xP", f"+{g1} xP")
    om = ", ".join(s_m["web_name"]) if not s_m.empty else "Hold"
    im = ", ".join(b_m["web_name"]) if not b_m.empty else "Hold"
    c3.metric(f"{max(2, fts)}-FT Swing: {om} → {im}", f"{xp_m} xP", f"+{gm} xP")

    view = st.radio("View Lineup For:", ["Current Squad (Roll)", "Best 1-FT Squad", f"Best {max(2, fts)}-FT Squad"], horizontal=True)
    act_sq = sq_1 if "1-FT" in view else (sq_m if "-FT" in view else curr_sq)
    ldf, sxp, bxp = solve_xi(act_sq, f"xp_{next_gw}")
    st.markdown(f"#### 🏆 Optimal GW{next_gw} Starting XI (**{sxp} xP** | Bench Boost Value: **+{bxp} xP**)")
    cols_show = ["web_name", "pos", "club", "price", "start", "cap", f"op_{next_gw}", f"xp_{next_gw}", "xp_5gw", "role"]
    st.dataframe(ldf.sort_values(["start", "element_type"], ascending=[False, True])[cols_show], use_container_width=True, hide_index=True)

with t2:
    st.markdown("#### 🪓 Squad Audit (Sorted Worst to Best over Next 5 GWs)")
    acols = ["web_name", "pos", "club", "price", "score", "verdict", "expected_goal_involvements_per_90", "defcon_90", "xp_5gw", "fix_5gw"]
    st.dataframe(curr_sq.sort_values("score", ascending=True)[acols], use_container_width=True, hide_index=True)

with t3:
    st.markdown("#### 🔍 Top Underlying & Fixture Targets (Next 5 GWs)")
    hcols = ["web_name", "pos", "club", "price", "selected_by_percent", "role", "score", "xp_5gw", "expected_goal_involvements_per_90", "defcon_90", "fix_5gw"]
    st.dataframe(p_df[p_df["minutes"] >= 180].sort_values("xp_5gw", ascending=False).head(50)[hcols], use_container_width=True, hide_index=True)
