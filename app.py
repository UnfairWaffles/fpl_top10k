import streamlit as st
import pandas as pd
import numpy as np
import requests
import pulp

st.set_page_config(
    page_title="Project Top 10k | FPL Optimizer",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------
# 1. LIVE FPL API DATA FETCHER & CACHE
# ---------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0"}

@st.cache_data(ttl=1800)
def fetch_fpl_core_data():
    boot_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    fix_url = "https://fantasy.premierleague.com/api/fixtures/"
    boot = requests.get(boot_url, headers=HEADERS, timeout=15).json()
    fixtures = requests.get(fix_url, headers=HEADERS, timeout=15).json()
    return boot, fixtures

def fetch_manager_data(manager_id, last_finished_gw):
    base = f"https://fantasy.premierleague.com/api/entry/{manager_id}"
    info_r = requests.get(f"{base}/", headers=HEADERS, timeout=10)
    if info_r.status_code != 200:
        return None
    info = info_r.json()
    hist = requests.get(f"{base}/history/", headers=HEADERS, timeout=10).json()
    transfers = requests.get(f"{base}/transfers/", headers=HEADERS, timeout=10).json()
    picks_gw = max(1, last_finished_gw)
    picks = requests.get(
        f"{base}/event/{picks_gw}/picks/", headers=HEADERS, timeout=10
    ).json()
    return {
        "info": info,
        "history": hist,
        "transfers": transfers,
        "picks": picks,
        "picks_gw": picks_gw,
    }

def calculate_banked_fts(history_json):
    """Reconstructs exact banked Free Transfers (1 to 5) from public history."""
    current = history_json.get("current", [])
    chips = {c["event"]: c["name"] for c in history_json.get("chips", [])}
    if not current:
        return 1
    ft = 1
    for i, gw_row in enumerate(current):
        ev = gw_row["event"]
        t_made = gw_row.get("event_transfers", 0)
        chip = chips.get(ev, None)
        if i == 0:
            ft = 1
            continue
        if chip in ["wildcard", "freehit"]:
            # 2024/25+ rule: WC and FH preserve banked FTs
            ft = min(5, max(1, ft))
        else:
            ft = min(5, max(0, ft - t_made) + 1)
    return int(np.clip(ft, 1, 5))

# ---------------------------------------------------------
# 2. UNDERLYING METRICS, DEFCON, BPS & FIXTURE ENGINE
# ---------------------------------------------------------
def build_player_model(boot, fixtures, next_gw, horizon=5, fixture_weight=0.45, mode="Balanced"):
    teams_df = pd.DataFrame(boot["teams"])
    players_df = pd.DataFrame(boot["elements"])
    fix_df = pd.DataFrame(fixtures)

    # Team lookup dictionaries
    team_short = dict(zip(teams_df["id"], teams_df["short_name"]))
    att_h = dict(zip(teams_df["id"], teams_df["strength_attack_home"]))
    att_a = dict(zip(teams_df["id"], teams_df["strength_attack_away"]))
    def_h = dict(zip(teams_df["id"], teams_df["strength_defence_home"]))
    def_a = dict(zip(teams_df["id"], teams_df["strength_defence_away"]))

    avg_att = np.mean(list(att_h.values()) + list(att_a.values()))
    avg_def = np.mean(list(def_h.values()) + list(def_a.values()))

    # Numeric conversions
    num_cols = [
        "now_cost", "selected_by_percent", "minutes", "total_points", "form",
        "points_per_game", "expected_goals_per_90", "expected_assists_per_90",
        "expected_goal_involvements_per_90", "expected_goals_conceded_per_90",
        "clean_sheets_per_90", "saves_per_90", "bps", "ict_index",
        "threat", "creativity"
    ]
    for col in num_cols:
        if col in players_df.columns:
            players_df[col] = pd.to_numeric(players_df[col], errors="coerce").fillna(0.0)
        else:
            players_df[col] = 0.0

    players_df["price"] = players_df["now_cost"] / 10.0
    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    players_df["pos"] = players_df["element_type"].map(pos_map)
    players_df["team_name"] = players_df["team"].map(team_short)

    # Minutes security & rotation factor
    games_played = max(1, next_gw - 1)
    max_possible_mins = games_played * 90.0
    players_df["mins_ratio"] = np.clip(players_df["minutes"] / max_possible_mins, 0.0, 1.0)
    chance_next = pd.to_numeric(
        players_df.get("chance_of_playing_next_round", 100), errors="coerce"
    ).fillna(100.0) / 100.0
    status_mult = players_df["status"].map(
        {"a": 1.0, "d": 0.75, "i": 0.0, "s": 0.0, "u": 0.0, "n": 0.0}
    ).fillna(1.0)
    players_df["mins_security"] = np.clip(
        (0.65 * players_df["mins_ratio"] + 0.35 * np.minimum(1.0, players_df["minutes"] / 180.0))
        * chance_next * status_mult,
        0.0, 1.0
    )

    # DefCon (+2 pts defensive contribution engine)
    if "defensive_contribution_per_90" in players_df.columns:
        raw_defcon = pd.to_numeric(
            players_df["defensive_contribution_per_90"], errors="coerce"
        ).fillna(0.0)
    elif "clearances_blocks_interceptions" in players_df.columns:
        cbi = pd.to_numeric(players_df["clearances_blocks_interceptions"], errors="coerce").fillna(0.0)
        tkl = pd.to_numeric(players_df.get("tackles", 0), errors="coerce").fillna(0.0)
        rec = pd.to_numeric(players_df.get("recoveries", 0), errors="coerce").fillna(0.0)
        nineties = np.maximum(1.0, players_df["minutes"] / 90.0)
        raw_defcon = (cbi + tkl + 0.5 * rec) / nineties
    else:
        nineties = np.maximum(1.0, players_df["minutes"] / 90.0)
        raw_defcon = (players_df["bps"] / nineties) * 0.35

    # Estimate probability of hitting +2pt DefCon threshold (10 for DEF, 12 for MID/FWD)
    def calc_defcon_exp_pts(row, dc_val):
        if row["pos"] == "DEF":
            hit_prob = np.clip(dc_val / 13.5, 0.0, 0.75)
            return hit_prob * 2.0
        elif row["pos"] in ["MID", "FWD"]:
            hit_prob = np.clip(dc_val / 16.5, 0.0, 0.65)
            return hit_prob * 2.0
        return 0.0

    players_df["defcon_per_90"] = raw_defcon
    players_df["defcon_xpts"] = [
        calc_defcon_exp_pts(row, dc) for (_, row), dc in zip(players_df.iterrows(), raw_defcon)
    ]

    # Set-Piece & Penalty Duty Boost
    pen_order = pd.to_numeric(players_df.get("penalties_order", 0), errors="coerce").fillna(0)
    cor_order = pd.to_numeric(
        players_df.get("corners_and_indirect_freekicks_order", 0), errors="coerce"
    ).fillna(0)
    fk_order = pd.to_numeric(players_df.get("direct_freekicks_order", 0), errors="coerce").fillna(0)

    def get_set_piece_tag(p, c, f):
        tags = []
        if p == 1:
            tags.append("PEN 1st")
        elif p == 2:
            tags.append("PEN 2nd")
        if c == 1:
            tags.append("COR 1st")
        if f == 1:
            tags.append("FK 1st")
        return ", ".join(tags) if tags else "-"

    players_df["set_pieces"] = [
        get_set_piece_tag(p, c, f) for p, c, f in zip(pen_order, cor_order, fk_order)
    ]
    players_df["sp_boost"] = (
        np.where(pen_order == 1, 0.45, np.where(pen_order == 2, 0.12, 0.0))
        + np.where(cor_order == 1, 0.20, 0.0)
        + np.where(fk_order == 1, 0.12, 0.0)
    )

    # 2026/27 BPS Rule Adjustment (Rewards ball-carrying attackers & high-save keepers)
    nineties = np.maximum(1.0, players_df["minutes"] / 90.0)
    bps_per_90 = players_df["bps"] / nineties
    players_df["bonus_xpts"] = np.clip((bps_per_90 - 12.0) / 14.0, 0.0, 2.2)
    players_df["bonus_xpts"] = np.where(
        players_df["pos"].isin(["MID", "FWD"]),
        players_df["bonus_xpts"] * 1.12,
        np.where(players_df["pos"] == "GKP", players_df["bonus_xpts"] * 1.10, players_df["bonus_xpts"] * 0.94),
    )

    # Base Per-90 Expected Points before fixture adjustments
    goal_pts = players_df["pos"].map({"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4})
    cs_pts = players_df["pos"].map({"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0})

    att_xpts = (
        players_df["expected_goals_per_90"] * goal_pts
        + players_df["expected_assists_per_90"] * 3.0
        + players_df["sp_boost"]
    )
    def_xpts = (
        np.clip(1.0 - (players_df["expected_goals_conceded_per_90"] / 2.2), 0.15, 0.60) * cs_pts
        + players_df["defcon_xpts"]
        + np.where(players_df["pos"] == "GKP", players_df["saves_per_90"] / 3.0, 0.0)
    )

    # Blend underlying xP with actual PPG/Form so proven clinical finishers aren't underscored
    raw_per90_xp = 2.0 + att_xpts + def_xpts + players_df["bonus_xpts"]
    blended_base_xp = (
        0.65 * raw_per90_xp
        + 0.20 * players_df["points_per_game"]
        + 0.15 * players_df["form"]
    )
    players_df["base_xp"] = np.clip(blended_base_xp, 0.5, 11.5) * players_df["mins_security"]

    # Shield vs Sword Ownership Classification
    players_df["role_tag"] = np.where(
        players_df["selected_by_percent"] >= 30.0,
        "🛡️ Shield (Template)",
        np.where(
            players_df["selected_by_percent"] <= 10.0,
            "⚔️ Sword (<10% Diff)",
            "⚖️ Core (10-30%)",
        ),
    )

    # Mode adjustment
    mode_mult = np.ones(len(players_df))
    if "Sword" in mode:
        mode_mult = np.where(players_df["selected_by_percent"] <= 10.0, 1.12, 0.96)
    elif "Shield" in mode:
        mode_mult = np.where(players_df["selected_by_percent"] >= 25.0, 1.10, 0.95)

    # Pre-build upcoming fixtures map per team across GW horizon
    gw_list = list(range(next_gw, min(39, next_gw + horizon)))
    future_fix = fix_df[fix_df["event"].isin(gw_list)]

    team_gw_fixtures = {t: {gw: [] for gw in gw_list} for t in teams_df["id"]}
    for _, f in future_fix.iterrows():
        gw = int(f["event"])
        th, ta = int(f["team_h"]), int(f["team_a"])
        # Attacking difficulty for home team depends on away defence strength
        att_mult_h = avg_def / max(800, def_a.get(ta, avg_def))
        def_mult_h = avg_att / max(800, att_a.get(ta, avg_att))
        team_gw_fixtures[th][gw].append({
            "opp": f"{team_short[ta]} (H)",
            "att_mult": att_mult_h * 1.06,
            "def_mult": def_mult_h * 1.06,
        })

        att_mult_a = avg_def / max(800, def_h.get(th, avg_def))
        def_mult_a = avg_att / max(800, att_h.get(th, avg_att))
        team_gw_fixtures[ta][gw].append({
            "opp": f"{team_short[th]} (A)",
            "att_mult": att_mult_a * 0.95,
            "def_mult": def_mult_a * 0.95,
        })

    # Compute GW-by-GW xP for every player
    for gw in gw_list:
        gw_xp_col = []
        gw_opp_col = []
        for _, row in players_df.iterrows():
            tid = int(row["team"])
            f_list = team_gw_fixtures.get(tid, {}).get(gw, [])
            if not f_list:
                gw_xp_col.append(0.0)
                gw_opp_col.append("BLANK")
                continue
            total_gw_p = 0.0
            opp_labels = []
            for fix_item in f_list:
                opp_labels.append(fix_item["opp"])
                if row["pos"] in ["GKP", "DEF"]:
                    f_mult = 0.70 * fix_item["def_mult"] + 0.30 * fix_item["att_mult"]
                else:
                    f_mult = 0.85 * fix_item["att_mult"] + 0.15 * fix_item["def_mult"]
                adj_mult = 1.0 + fixture_weight * (f_mult - 1.0)
                total_gw_p += row["base_xp"] * adj_mult
            gw_xp_col.append(round(total_gw_p * mode_mult[row.name], 2))
            gw_opp_col.append(" + ".join(opp_labels))
        players_df[f"xp_gw{gw}"] = gw_xp_col
        players_df[f"opp_gw{gw}"] = gw_opp_col

    # Next 5 fixture string & 5-GW total xP
    xp_cols = [f"xp_gw{gw}" for gw in gw_list]
    opp_cols = [f"opp_gw{gw}" for gw in gw_list]
    players_df["total_5gw_xp"] = players_df[xp_cols].sum(axis=1).round(2)
    players_df["next_5_fixtures"] = players_df[opp_cols].apply(lambda r: " → ".join(r.values), axis=1)

    # Composite 0-100 Rating & Chopping Block Alert Logic
    max_5gw = max(1.0, players_df["total_5gw_xp"].quantile(0.98))
    players_df["rating_100"] = np.clip((players_df["total_5gw_xp"] / max_5gw) * 100.0, 0.0, 99.0).round(1)

    def diagnose_player(row):
        reasons = []
        if row["mins_security"] < 0.65:
            reasons.append("⚠️ Minutes/Injury Risk (<65m)")
        if row[f"xp_gw{gw_list[0]}"] < 3.2 and row["total_5gw_xp"] < 16.5:
            reasons.append("🧱 Fixture Cliff / Low Ceiling")
        if (
            row["pos"] in ["MID", "FWD"]
            and row["form"] > 5.0
            and row["expected_goal_involvements_per_90"] < 0.25
        ):
            reasons.append("📉 xGI Mirage (Unsustainable Form)")
        if not reasons:
            if row["rating_100"] >= 72:
                return "✅ STRONG HOLD / LOCK"
            return "⚖️ HOLD / BENCHABLE"
        return "🚨 SELL ALERT: " + " | ".join(reasons)

    players_df["verdict"] = players_df.apply(diagnose_player, axis=1)
    return players_df, gw_list

# ---------------------------------------------------------
# 3. MILP SOLVER (STARTING XI, TRANSFERS & ROLL PLANNER)
# ---------------------------------------------------------
def optimize_single_gw_lineup(squad_df, gw_col):
    """Picks valid optimal Starting XI (1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD), Captain, and ordered Bench."""
    df = squad_df.copy().reset_index(drop=True)
    prob = pulp.LpProblem("Lineup_Opt", pulp.LpMaximize)
    n = len(df)
    start = [pulp.LpVariable(f"s_{i}", cat="Binary") for i in range(n)]
    cap = [pulp.LpVariable(f"c_{i}", cat="Binary") for i in range(n)]

    xp = df[gw_col].values
    prob += pulp.lpSum([xp[i] * (start[i] + cap[i]) for i in range(n)])

    prob += pulp.lpSum(start) == 11
    prob += pulp.lpSum(cap) == 1
    for i in range(n):
        prob += cap[i] <= start[i]

    for pos, lo, hi in [("GKP", 1, 1), ("DEF", 3, 5), ("MID", 2, 5), ("FWD", 1, 3)]:
        idx = [i for i in range(n) if df.loc[i, "pos"] == pos]
        prob += pulp.lpSum([start[i] for i in idx]) >= lo
        prob += pulp.lpSum([start[i] for i in idx]) <= hi

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    df["is_starter"] = [int(pulp.value(start[i]) or 0) for i in range(n)]
    df["is_captain"] = [int(pulp.value(cap[i]) or 0) for i in range(n)]
    starters_xp = sum(xp[i] * (df.loc[i, "is_starter"] + df.loc[i, "is_captain"]) for i in range(n))
    bench_xp = sum(xp[i] * (1 - df.loc[i, "is_starter"]) for i in range(n))
    return df, round(starters_xp, 2), round(bench_xp, 2)

def find_best_transfers(all_players_df, current_ids, bank_m, max_transfers, gw_list, gamma=0.90, locked_ids=None, force_sell_ids=None):
    """Uses PuLP MILP to find the optimal K-transfer package over the 5-GW discounted horizon."""
    locked_ids = set(locked_ids or [])
    force_sell_ids = set(force_sell_ids or [])

    # Pre-filter candidate pool to top 160 players + current 15 squad for instant solve speed
    current_set = set(current_ids)
    top_pool = set(all_players_df.sort_values("total_5gw_xp", ascending=False).head(160)["id"])
    pool_df = all_players_df[all_players_df["id"].isin(current_set.union(top_pool))].copy().reset_index(drop=True)

    # Discounted multi-GW objective per player
    weights = [gamma ** idx for idx in range(len(gw_list))]
    disc_xp = np.zeros(len(pool_df))
    for w, gw in zip(weights, gw_list):
        disc_xp += w * pool_df[f"xp_gw{gw}"].values
    pool_df["disc_xp"] = disc_xp

    n = len(pool_df)
    in_squad = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    start = [pulp.LpVariable(f"s_{i}", cat="Binary") for i in range(n)]
    cap = [pulp.LpVariable(f"c_{i}", cat="Binary") for i in range(n)]

    prob = pulp.LpProblem("Transfer_Planner", pulp.LpMaximize)
    gw1_xp = pool_df[f"xp_gw{gw_list[0]}"].values

    # Maximize immediate GW1 starting XI + captain + multi-week discounted squad value
    prob += pulp.lpSum([
        0.55 * gw1_xp[i] * (start[i] + cap[i]) + 0.45 * pool_df.loc[i, "disc_xp"] * (start[i] + 0.15 * in_squad[i])
        for i in range(n)
    ])

    # Squad composition rules (2 GKP, 5 DEF, 5 MID, 3 FWD = 15 total)
    prob += pulp.lpSum(in_squad) == 15
    prob += pulp.lpSum(start) == 11
    prob += pulp.lpSum(cap) == 1

    for i in range(n):
        prob += start[i] <= in_squad[i]
        prob += cap[i] <= start[i]
        pid = int(pool_df.loc[i, "id"])
        if pid in locked_ids:
            prob += in_squad[i] == 1
        if pid in force_sell_ids and pid in current_set:
            prob += in_squad[i] == 0

    for pos, tot_req, lo, hi in [("GKP", 2, 1, 1), ("DEF", 5, 3, 5), ("MID", 5, 2, 5), ("FWD", 3, 1, 3)]:
        idx = [i for i in range(n) if pool_df.loc[i, "pos"] == pos]
        prob += pulp.lpSum([in_squad[i] for i in idx]) == tot_req
        prob += pulp.lpSum([start[i] for i in idx]) >= lo
        prob += pulp.lpSum([start[i] for i in idx]) <= hi

    # Max 3 players per Premier League club
    for tid in pool_df["team"].unique():
        idx = [i for i in range(n) if pool_df.loc[i, "team"] == tid]
        prob += pulp.lpSum([in_squad[i] for i in idx]) <= 3

    # Budget constraint
    current_squad_cost = pool_df[pool_df["id"].isin(current_set)]["price"].sum()
    total_budget = current_squad_cost + bank_m
    prob += pulp.lpSum([pool_df.loc[i, "price"] * in_squad[i] for i in range(n)]) <= total_budget + 0.001

    # Transfer count constraint
    kept_indices = [i for i in range(n) if int(pool_df.loc[i, "id"]) in current_set]
    prob += pulp.lpSum([in_squad[i] for i in kept_indices]) >= 15 - max_transfers

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    chosen_idx = [i for i in range(n) if (pulp.value(in_squad[i]) or 0) > 0.5]
    new_squad = pool_df.loc[chosen_idx].copy()
    new_ids = set(new_squad["id"])

    sold_df = pool_df[pool_df["id"].isin(current_set - new_ids)].copy()
    bought_df = pool_df[pool_df["id"].isin(new_ids - current_set)].copy()

    # Evaluate resulting squad across all 5 GWs
    total_5gw_lineup_xp = 0.0
    gw_breakdown = []
    for idx_g, gw in enumerate(gw_list):
        _, s_xp, b_xp = optimize_single_gw_lineup(new_squad, f"xp_gw{gw}")
        total_5gw_lineup_xp += (gamma ** idx_g) * s_xp
        gw_breakdown.append({"GW": f"GW{gw}", "Starters_xP": s_xp, "Bench_xP": b_xp})

    return new_squad, sold_df, bought_df, round(total_5gw_lineup_xp, 2), gw_breakdown

# ---------------------------------------------------------
# 4. STREAMLIT USER INTERFACE
# ---------------------------------------------------------
st.title("⚽ Project Top 10k: FPL Command Centre")
st.caption("Powered by Opta Underlying xGI, DefCon (+2pt) Hit Rates, 2026/27 BPS Rules & 1–5 Free Transfer MILP Solver")

boot, fixtures = fetch_fpl_core_data()
events = boot["events"]
next_event = next((e for e in events if e.get("is_next")), events[-1])
curr_event = next((e for e in events if e.get("is_current")), next_event)
next_gw = int(next_event["id"])
last_finished_gw = max(1, next_gw - 1)

with st.sidebar:
    st.header("1. Your FPL Team Setup")
    manager_id_input = st.text_input("Enter Your FPL Manager ID", value="", placeholder="e.g. 1234567")
    with st.expander("Where do I find my Manager ID?"):
        st.write(
            "Open the FPL website in your browser, click **Points** or **Pick Team**, and look at the web address bar. "
            "The number right after `/entry/` (e.g., `fantasy.premierleague.com/entry/1234567/event/6`) is your ID."
        )

    st.header("2. Strategy & Model Weights")
    strategy_mode = st.selectbox(
        "Rank Strategy Mode",
        ["Balanced (Max Points)", "Sword (Hunt Differentials <10% Owned)", "Shield (Protect Rank / High EO)"],
    )
    fixture_weight = st.slider("Fixture Difficulty Weighting", 0.10, 0.85, 0.45, 0.05)
    gamma_decay = st.slider("Future Week Confidence (Horizon Decay)", 0.75, 1.00, 0.90, 0.05)

players_df, gw_list = build_player_model(
    boot, fixtures, next_gw=next_gw, horizon=5, fixture_weight=fixture_weight, mode=strategy_mode
)

# Load manager squad if ID entered, otherwise build a smart sample squad so the app works immediately
manager_loaded =