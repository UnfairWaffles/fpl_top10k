import streamlit as st, pandas as pd, numpy as np, requests

H = {"User-Agent": "Mozilla/5.0"}

@st.cache_data(ttl=1800)
def load_fpl():
    b = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=H, timeout=15).json()
    f = requests.get("https://fantasy.premierleague.com/api/fixtures/", headers=H, timeout=15).json()
    return b, f

@st.cache_data(ttl=600)
def get_mgr(mid, gw):
    u = f"https://fantasy.premierleague.com/api/entry/{mid}"
    r = requests.get(f"{u}/", headers=H, timeout=10)
    if r.status_code != 200: return None
    h = requests.get(f"{u}/history/", headers=H, timeout=10).json()
    t = requests.get(f"{u}/transfers/", headers=H, timeout=10).json()
    p = requests.get(f"{u}/event/{max(1, gw)}/picks/", headers=H, timeout=10).json()
    return {"info": r.json(), "hist": h, "tr": t, "picks": p}

@st.cache_data(ttl=900)
def get_league(lid, gw):
    u = f"https://fantasy.premierleague.com/api/leagues-classic/{lid}/standings/"
    r = requests.get(u, headers=H, timeout=10)
    if r.status_code != 200: return None
    data = r.json()
    top5 = data.get("standings", {}).get("results", [])[:5]
    rivals = []
    for m in top5:
        eid = m["entry"]
        p = requests.get(f"https://fantasy.premierleague.com/api/entry/{eid}/event/{max(1, gw)}/picks/", headers=H, timeout=10).json()
        picks = p.get("picks", [])
        cap = next((x["element"] for x in picks if x.get("is_captain")), None)
        rivals.append({"name": m["entry_name"], "mgr": m["player_name"], "pts": m["total"], "ids": [x["element"] for x in picks], "cap": cap})
    return {"name": data.get("league", {}).get("name", "Mini-League"), "rivals": rivals}

def calc_fts(hist):
    chips = {c["event"]: c["name"] for c in hist.get("chips", [])}
    ft = 1
    for i, r in enumerate(hist.get("current", [])):
        if i == 0: ft = 1
        elif chips.get(r["event"]) in ["wildcard", "freehit"]: ft = min(5, max(1, ft))
        else: ft = min(5, max(0, ft - r.get("event_transfers", 0)) + 1)
    return int(np.clip(ft, 1, 5))

def apply_selling_prices(p_df, sq_ids, tr_list):
    d = p_df.copy()
    buy_map = {}
    for t in reversed(tr_list or []):
        buy_map[t["element_in"]] = t["element_in_cost"] / 10.0
    sell_p, locked_v = [], []
    for _, r in d.iterrows():
        pid, now_p = int(r["id"]), float(r["price"])
        if pid in sq_ids:
            bp = buy_map.get(pid, round(now_p - float(r.get("cost_change_start", 0))/10.0, 1))
            if now_p > bp:
                gain = int(round((now_p - bp)*10)) // 2
                sp = round(bp + gain / 10.0, 1)
            else: sp = now_p
            sell_p.append(sp)
            locked_v.append(round(now_p - sp, 1))
        else:
            sell_p.append(now_p); locked_v.append(0.0)
    d["sell_price"], d["locked_val"] = sell_p, locked_v
    return d

def build_df(b, f, nxt, fix_w, mode):
    t_df, p_df, f_df = pd.DataFrame(b["teams"]), pd.DataFrame(b["elements"]), pd.DataFrame(f)
    gws = list(range(nxt, min(39, nxt + 5)))
    t_map = dict(zip(t_df["id"], t_df["short_name"]))
    ah, aa = dict(zip(t_df["id"], t_df["strength_attack_home"])), dict(zip(t_df["id"], t_df["strength_attack_away"]))
    dh, da = dict(zip(t_df["id"], t_df["strength_defence_home"])), dict(zip(t_df["id"], t_df["strength_defence_away"]))
    avg_s = float(np.mean(list(ah.values()) + list(dh.values())))

    cols = ["now_cost","cost_change_start","selected_by_percent","minutes","form","points_per_game","expected_goals_per_90","expected_assists_per_90","expected_goal_involvements_per_90","expected_goals_conceded_per_90","saves_per_90","bps","penalties_order","corners_and_indirect_freekicks_order","transfers_in_event","transfers_out_event"]
    for c in cols: p_df[c] = pd.to_numeric(p_df[c], errors="coerce").fillna(0.0) if c in p_df.columns else 0.0

    p_df["price"] = p_df["now_cost"] / 10.0
    p_df["sell_price"] = p_df["price"]
    p_df["locked_val"] = 0.0
    p_df["pos"] = p_df["element_type"].map({1:"GKP", 2:"DEF", 3:"MID", 4:"FWD"})
    p_df["club"] = p_df["team"].map(t_map)

    gms = max(1, nxt - 1)
    ch = pd.to_numeric(p_df["chance_of_playing_next_round"], errors="coerce").fillna(100.0)/100.0 if "chance_of_playing_next_round" in p_df.columns else 1.0
    st_m = p_df["status"].map({"a":1.0, "d":0.75}).fillna(0.0)
    p_df["mins_sec"] = np.clip((0.65*np.clip(p_df["minutes"]/(gms*90.0),0,1) + 0.35*np.minimum(1.0, p_df["minutes"]/180.0))*ch*st_m, 0.0, 1.0)

    n90 = np.maximum(1.0, p_df["minutes"] / 90.0)
    dc = pd.to_numeric(p_df["defensive_contribution_per_90"], errors="coerce").fillna(0.0) if "defensive_contribution_per_90" in p_df.columns else (p_df["bps"]/n90)*0.38
    p_df["defcon_90"] = dc.round(2)
    dc_pts = np.where(p_df["pos"]=="DEF", np.clip(dc/13.5,0,0.75)*2.0, np.where(p_df["pos"].isin(["MID","FWD"]), np.clip(dc/16.5,0,0.65)*2.0, 0.0))

    sp = np.where(p_df["penalties_order"]==1, 0.45, 0.0) + np.where(p_df["corners_and_indirect_freekicks_order"]==1, 0.20, 0.0)
    gp = p_df["pos"].map({"GKP":6, "DEF":6, "MID":5, "FWD":4})
    cp = p_df["pos"].map({"GKP":4, "DEF":4, "MID":1, "FWD":0})
    raw_xp = 2.0 + p_df["expected_goals_per_90"]*gp + p_df["expected_assists_per_90"]*3.0 + sp + np.clip(1.0 - p_df["expected_goals_conceded_per_90"]/2.2, 0.15, 0.60)*cp + dc_pts + np.clip(((p_df["bps"]/n90)-12.0)/14.0, 0, 2.0) + np.where(p_df["pos"]=="GKP", p_df["saves_per_90"]/3.0, 0.0)

    # Feature 5: Last-4-GW Form Momentum Surge
    p_df["momentum"] = (p_df["form"] - p_df["points_per_game"]).round(2)
    p_df["surge_tag"] = np.where(p_df["momentum"] >= 1.5, "🔥 Hot Streak", np.where(p_df["momentum"] <= -1.5, "❄️ Cold", "➖ Steady"))
    p_df["base_xp"] = np.clip(0.60*raw_xp + 0.20*p_df["points_per_game"] + 0.20*p_df["form"], 0.5, 11.5) * p_df["mins_sec"]

    # Feature 2: Live Price Velocity
    p_df["net_tr"] = (p_df["transfers_in_event"] - p_df["transfers_out_event"]).astype(int)
    p_df["price_trend"] = np.where(p_df["net_tr"] > 45000, "📈 Rising Tonight", np.where(p_df["net_tr"] < -45000, "📉 Falling Tonight", "➖ Stable"))

    p_df["role"] = np.where(p_df["selected_by_percent"]>=30, "🛡️ Shield", np.where(p_df["selected_by_percent"]<=10, "⚔️ Sword", "⚖️ Core"))
    m_mult = np.where(p_df["selected_by_percent"]<=10, 1.12, 0.96) if "Sword" in mode else (np.where(p_df["selected_by_percent"]>=25, 1.10, 0.95) if "Shield" in mode else np.ones(len(p_df)))

    t_fix = {int(t): {g: [] for g in gws} for t in t_df["id"]}
    for _, r in f_df[f_df["event"].isin(gws)].iterrows():
        g, th, ta = int(r["event"]), int(r["team_h"]), int(r["team_a"])
        t_fix[th][g].append((f"{t_map[ta]}(H)", (avg_s/max(800.0, float(da[ta])))*1.06, (avg_s/max(800.0, float(aa[ta])))*1.06))
        t_fix[ta][g].append((f"{t_map[th]}(A)", (avg_s/max(800.0, float(dh[th])))*0.95, (avg_s/max(800.0, float(ah[th])))*0.95))

    for g in gws:
        xl, ol = [], []
        for i, r in p_df.iterrows():
            fl = t_fix.get(int(r["team"]), {}).get(g, [])
            if not fl: xl.append(0.0); ol.append("BLANK"); continue
            tot = sum(float(r["base_xp"])*(1.0 + fix_w*((0.7*dm+0.3*am if r["pos"] in ["GKP","DEF"] else 0.85*am+0.15*dm) - 1.0)) for _, am, dm in fl)
            xl.append(round(tot * float(m_mult[i]), 2)); ol.append("+".join(op for op, _, _ in fl))
        p_df[f"xp_{g}"], p_df[f"op_{g}"] = xl, ol

    p_df["xp_5gw"] = p_df[[f"xp_{g}" for g in gws]].sum(axis=1).round(2)
    p_df["disc"] = sum((0.9**i)*p_df[f"xp_{g}"].values for i, g in enumerate(gws))
    p_df["fix_5gw"] = p_df[[f"op_{g}" for g in gws]].apply(lambda r: " → ".join(r.values), axis=1)
    p_df["score"] = np.clip((p_df["xp_5gw"] / max(1.0, float(p_df["xp_5gw"].quantile(0.98))))*100.0, 0, 99).round(1)
    p_df["verdict"] = np.where(p_df["mins_sec"]<0.65, "🚨 SELL: Mins Risk", np.where(p_df["xp_5gw"]<16.5, "🚨 SELL: Fixture Cliff", np.where(p_df["score"]>=72, "✅ STRONG HOLD", "⚖️ HOLD")))
    return p_df, gws

def solve_xi(df, col):
    d = df.copy().reset_index(drop=True)
    d["start"], d["cap"] = 0, 0
    idx = []
    for pos, k in [("GKP",1), ("DEF",3), ("MID",2), ("FWD",1)]:
        idx.extend(d[d["pos"]==pos].sort_values(col, ascending=False).head(k).index.tolist())
    idx.extend(d[(d["pos"]!="GKP") & (~d.index.isin(idx))].sort_values(col, ascending=False).head(11-len(idx)).index.tolist())
    d.loc[idx, "start"] = 1
    st_idx = d[d["start"]==1].sort_values(col, ascending=False).index
    if len(st_idx) > 0: d.loc[st_idx[0], "cap"] = 1
    return d, round(float((d[col]*(d["start"]+d["cap"])).sum()), 2), round(float((d[col]*(1-d["start"])).sum()), 2)

def eval_5gw(df, gws): return round(sum((0.9**i)*solve_xi(df, f"xp_{g}")[1] for i, g in enumerate(gws)), 2)

def solve_2ft_combo(all_d, c_ids, bnk, gws):
    """Feature 4: Simultaneous 2-Player Downgrade + Upgrade Pair Finder."""
    w_sq = all_d[all_d["id"].isin(c_ids)].copy().reset_index(drop=True)
    c_set = set(c_ids)
    pool = all_d[~all_d["id"].isin(c_set)].sort_values("disc", ascending=False).groupby("pos").head(12)
    best_gain, best_pair = -999.0, None
    n = len(w_sq)
    for i in range(n):
        for j in range(i+1, n):
            o1, o2 = w_sq.iloc[i], w_sq.iloc[j]
            avail = float(o1["sell_price"]) + float(o2["sell_price"]) + float(bnk) + 0.001
            rem_clubs = w_sq.drop([i, j])["team"].value_counts().to_dict()
            c1s = pool[pool["pos"]==o1["pos"]]
            c2s = pool[pool["pos"]==o2["pos"]]
            for _, i1 in c1s.iterrows():
                for _, i2 in c2s.iterrows():
                    if int(i1["id"]) == int(i2["id"]): continue
                    if float(i1["price"]) + float(i2["price"]) > avail: continue
                    t1, t2 = int(i1["team"]), int(i2["team"])
                    if rem_clubs.get(t1, 0) + (1 if t1==t2 else 0) + 1 > 3: continue
                    if rem_clubs.get(t2, 0) + 1 > 3: continue
                    gn = float(i1["disc"]) + float(i2["disc"]) - float(o1["disc"]) - float(o2["disc"])
                    if gn > best_gain:
                        best_gain = gn; best_pair = (o1, o2, i1, i2)
    if best_pair and best_gain > 0:
        o1, o2, i1, i2 = best_pair
        new_sq = pd.concat([w_sq[~w_sq["id"].isin([int(o1["id"]), int(o2["id"])])], pd.DataFrame([i1, i2])], ignore_index=True)
        return new_sq, [o1["web_name"], o2["web_name"]], [i1["web_name"], i2["web_name"]], eval_5gw(new_sq, gws)
    return w_sq, [], [], eval_5gw(w_sq, gws)

def build_wildcard(all_d, budget, col, gws):
    """Feature 6: Optimal 15-Man Wildcard / Free Hit Builder under budget & 3-per-club rules."""
    pool = all_d[all_d["minutes"]>=90].copy()
    pool["val_score"] = pool[col] * 0.75 + (pool[col] / np.maximum(4.0, pool["price"])) * 4.0
    pool = pool.sort_values("val_score", ascending=False)
    req = {"GKP":2, "DEF":5, "MID":5, "FWD":3}
    chosen, clubs, spent = [], {}, 0.0
    for _, r in pool.iterrows():
        p, t, pr = r["pos"], int(r["team"]), float(r["price"])
        if req[p] > 0 and clubs.get(t, 0) < 3:
            rem_slots = sum(req.values()) - 1
            if spent + pr + rem_slots * 4.0 <= budget + 0.1:
                chosen.append(r); req[p] -= 1; clubs[t] = clubs.get(t, 0) + 1; spent += pr
    wc_df = pd.DataFrame(chosen)
    return wc_df, round(spent, 1), eval_5gw(wc_df, gws)
# --- END OF ENGINE ---
