import streamlit as st
import pandas as pd
import numpy as np
import requests

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
    avg_s = float(np.mean(list(att_h.values()) + list(def_h.values())))

    num_cols = [
        "now_cost", "selected_by_percent", "minutes", "form", "points_per_game",
        "expected_goals_per_90", "expected_assists_per_90",
        "expected_goal_involvements_per_90", "expected_goals_conceded_per_90",
        "saves_per_90", "bps", "penalties_order", "corners_and_indirect_freekicks_order"
    ]
    for col_name in num_cols:
        if col_name in p_df.columns:
            p_df[col_name] = pd.to_numeric(p_df[col_name], errors="coerce").fillna(0.0)
        else:
            p_df[col_name] = 0.0

    p_df["price"] = p_df["now_cost"] / 10.0
    p_df["pos"] = p_df["element_type"].map({1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"})
    p_df["club"] = p_df["team"].map(t_short)

    games = max(1, next_gw - 1)
    m_ratio = np.clip(p_df["minutes"] / (games * 90.0), 0.0, 1.0)
    chance = pd.to_numeric(p_df.get("chance_of_playing_next_round", 100), errors="coerce").fillna(100.0) / 100.0 if "chance_of_playing_next_round" in p_df.columns else 1.0
    stat_m = p_df["status"].map({"a": 1.0, "d": 0.75}).fillna(0.0)
    p_df["mins_sec"] = np.clip((0.65 * m_ratio + 0.35 * np.minimum(1.0, p_df["minutes"] / 180.0)) * chance * stat_m, 0.0, 1.0)

    nineties = np.maximum(1.0, p_df["minutes"] / 90.0)
    dc = pd.to_numeric(p_df["defensive_contribution_per_90"], errors="coerce").fillna(0.0) if "defensive_contribution_per_90" in p_df.columns else (p_df["bps"] / nineties) * 0.38
    p_df["defcon_90"] = dc.round(2)
    p_df["defcon_pts"] = np.where(
        p_df["pos"] == "DEF", np.clip(dc / 13.5, 0.0, 0.75) * 2.0,
        np.where(p_df["pos"].isin(["MID", "FWD"]), np.clip(dc / 16.5, 0.0, 0.65) * 2.0, 0.0)
    )

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
    t_fix = {int(t): {int(g): [] for g in gws} for t in t_df["id"]}

    for _, f in f_sub.iterrows():
        if pd.isna(f["event"]):
            continue
        g, th, ta = int(f["event"]), int(f["team_h"]), int(f["team_a"])
        if g in gws:
            t_fix[th][g].append((f"{t_short[ta]}(H)", (avg_s / max(800.0, float(def_a[ta]))) * 1.06, (avg_s / max(800.0, float(att_a[ta]))) * 1.06))
            t_fix[ta][g].append((f"{t_short[th]}(A)", (avg_s / max(800.0, float(def_h[th]))) * 0.95, (avg_s / max(800.0, float(att_h[th]))) * 0.95))

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
                tot += float(r["base_xp"]) * (1.0 + fix_w * (fm - 1.0))
            xp_l.append(round(tot * float(m_mult[idx]), 2))
            op_l.append("+".join(ops))
        p_df[f"xp_{g}"] = xp_l
        p_df[f"op_{g}"] = op_l

    p_df["xp_5gw"] = p_df[[f"xp_{g}" for g in gws]].sum(axis=1).round(2)
    p_df["fix_5gw"] = p_df[[f"op_{g}" for g in gws]].apply(lambda r: " → ".join(r.values), axis=1)
    q98 = max(1.0, float(p_df["xp_5gw"].quantile(0.98)))
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
    df["start"] = 0
    df["cap"] = 0

    gkp = df[df["pos"] == "GKP"].sort_values(gw_col, ascending=False)
    defs = df[df["pos"] == "DEF"].sort_values(gw_col, ascending=False)
    mids = df[df["pos"] == "MID"].sort_values(gw_col, ascending=False)
    fwds = df[df["pos"] == "FWD"].sort_values(gw_col, ascending=False)

    locked_idx = []
    if not gkp.empty:
        locked_idx.append(gkp.index[0])
    locked_idx.extend(defs.head(3).index.tolist())
    locked_idx.extend(mids.head(2).index.tolist())
    locked_idx.extend(fwds.head(1).index.tolist())

    rem_outfield = df[(df["pos"] != "GKP") & (~df.index.isin(locked_idx))].sort_values(gw_col, ascending=False)
    locked_idx.extend(rem_outfield.head(11 - len(locked_idx)).index.tolist())

    df.loc[locked_idx, "start"] = 1
    starters = df[df["start"] == 1].sort_values(gw_col, ascending=False)
    if not starters.empty:
        df.loc[starters.index[0], "cap"] = 1

    s_xp = float((df[gw_col] * (df["start"] + df["cap"])).sum())
    b_xp = float((df[gw_col] * (1 - df["start"])).sum())
    return df, round(s_xp, 2), round(b_xp, 2)

def eval_squad_5gw(sq_df, gws, gamma=0.90):
    return round(sum((gamma ** i) * solve_xi(sq_df, f"xp_{g}")[1] for i, g in enumerate(gws)), 2)

def solve_transfers(all_df, curr_ids, bank, max_t, gws, gamma=0.90):
    # Compute 'disc' on all_df FIRST so both candidate pool and current squad have the column
    weights = np.array([gamma ** i for i in range(len(gws))])
    all_df = all_df.copy().reset_index(drop=True)
    all_df["disc"] = sum(w * all_df[f"xp_{g}"].values for w, g in zip(weights, gws))

    curr_sq = all_df[all_df["id"].isin(curr_ids)].copy().reset_index(drop=True)
    base_xp = eval_squad_5gw(curr_sq, gws, gamma)
    if max_t <= 0:
        return curr_sq, pd.DataFrame(), pd.DataFrame(), base_xp

    working_sq = curr_sq.copy().reset_index(drop=True)
    rem_bank = float(bank)
    sold_list, bought_list = [], []
    orig_ids = set(curr_ids)

    for _ in range(int(max_t)):
        best_gain = 0.0
        best_out_id, best_in_row = None, None
        current_id_set = set(working_sq["id"].values)
        club_counts = working_sq["team"].value_counts().to_dict()

        for _, out_row in working_sq.iterrows():
            out_id = int(out_row["id"])
            if out_id not in orig_ids:
                continue
            pos = out_row["pos"]
            max_afford = float(out_row["price"]) + rem_bank + 0.001
            out_team = int(out_row["team"])
            out_disc = float(out_row["disc"])

            cands = all_df[
                (all_df["pos"] == pos)
                & (~all_df["id"].isin(current_id_set))
                & (all_df["price"] <= max_afford)
            ].sort_values("disc", ascending=False).head(15)

            for _, in_row in cands.iterrows():
                in_team = int(in_row["team"])
                if in_team != out_team and club_counts.get(in_team, 0) >= 3:
                    continue
                gain = float(in_row["disc"]) - out_disc
                if gain > best_gain:
                    best_gain = gain
                    best_out_id = out_id
                    best_in_row = in_row

        if best_out_id is not None and best_in_row is not None:
            out_r = working_sq[working_sq["id"] == best_out_id].iloc[0]
            sold_list.append(out_r)
            bought_list.append(best_in_row)
            rem_bank += float(out_r["price"]) - float(best_in_row["price"])
            working_sq = pd.concat(
                [working_sq[working_sq["id"] != best_out_id], pd.DataFrame([best_in_row])],
                ignore_index=True,
            )
        else:
            break

    sold_df = pd.DataFrame(sold_list)
    bought_df = pd.DataFrame(bought_list)
    tot_xp = eval_squad_5gw(working_sq, gws, gamma)
    return working_sq, sold_df, bought_df, tot_xp

st.title("⚽ Project Top 10k
