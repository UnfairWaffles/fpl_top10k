import streamlit as st, pandas as pd, numpy as np, requests

st.set_page_config(page_title="Project Top 10k", page_icon="⚽", layout="wide")
H = {"User-Agent": "Mozilla/5.0"}

@st.cache_data(ttl=1800)
def load_fpl():
    b = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=H, timeout=15).json()
    f = requests.get("https://fantasy.premierleague.com/api/fixtures/", headers=H, timeout=15).json()
    return b, f

def get_mgr(mid, gw):
    u = f"https://fantasy.premierleague.com/api/entry/{mid}"
    r = requests.get(f"{u}/", headers=H, timeout=10)
    if r.status_code != 200: return None
    h = requests.get(f"{u}/history/", headers=H, timeout=10).json()
    p = requests.get(f"{u}/event/{max(1, gw)}/picks/", headers=H, timeout=10).json()
    return {"info": r.json(), "hist": h, "picks": p}

def calc_fts(hist):
    chips = {c["event"]: c["name"] for c in hist.get("chips", [])}
    ft = 1
    for i, r in enumerate(hist.get("current", [])):
        if i == 0: ft = 1
        elif chips.get(r["event"]) in ["wildcard", "freehit"]: ft = min(5, max(1, ft))
        else: ft = min(5, max(0, ft - r.get("event_transfers", 0)) + 1)
    return int(np.clip(ft, 1, 5))

b, f = load_fpl()
nxt = int(next((e["id"] for e in b["events"] if e.get("is_next")), b["events"][-1]["id"]))
gws = list(range(nxt, min(39, nxt + 5)))

with st.sidebar:
    st.header("⚙️ Team & Model Settings")
    mid_in = st.text_input("FPL Manager ID", "", placeholder="e.g. 1234567")
    mode = st.selectbox("Strategy Mode", ["Balanced", "Sword (<10% Diffs)", "Shield (High EO)"])
    fix_w = st.slider("Fixture Weighting", 0.10, 0.85, 0.45, 0.05)

t_df, p_df, f_df = pd.DataFrame(b["teams"]), pd.DataFrame(b["elements"]), pd.DataFrame(f)
t_map = dict(zip(t_df["id"], t_df["short_name"]))
ah, aa = dict(zip(t_df["id"], t_df["strength_attack_home"])), dict(zip(t_df["id"], t_df["strength_attack_away"]))
dh, da = dict(zip(t_df["id"], t_df["strength_defence_home"])), dict(zip(t_df["id"], t_df["strength_defence_away"]))
avg_s = float(np.mean(list(ah.values()) + list(dh.values())))

cols = ["now_cost","selected_by_percent","minutes","form","points_per_game","expected_goals_per_90","expected_assists_per_90","expected_goal_involvements_per_90","expected_goals_conceded_per_90","saves_per_90","bps","penalties_order","corners_and_indirect_freekicks_order"]
for c in cols: p_df[c] = pd.to_numeric(p_df[c], errors="coerce").fillna(0.0) if c in p_df.columns else 0.0

p_df["price"] = p_df["now_cost"] / 10.0
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
p_df["base_xp"] = np.clip(0.65*raw_xp + 0.20*p_df["points_per_game"] + 0.15*p_df["form"], 0.5, 11.5) * p_df["mins_sec"]
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
        xl.append(round(tot * float(m_mult[i]), 2))
        ol.append("+".join(op for op, _, _ in fl))
    p_df[f"xp_{g}"], p_df[f"op_{g}"] = xl, ol

p_df["xp_5gw"] = p_df[[f"xp_{g}" for g in gws]].sum(axis=1).round(2)
p_df["disc"] = sum((0.9**i)*p_df[f"xp_{g}"].values for i, g in enumerate(gws))
p_df["fix_5gw"] = p_df[[f"op_{g}" for g in gws]].apply(lambda r: " → ".join(r.values), axis=1)
p_df["score"] = np.clip((p_df["xp_5gw"] / max(1.0, float(p_df["xp_5gw"].quantile(0.98))))*100.0, 0, 99).round(1)
p_df["verdict"] = np.where(p_df["mins_sec"]<0.65, "🚨 SELL: Mins Risk", np.where(p_df["xp_5gw"]<16.5, "🚨 SELL: Fixture Cliff", np.where(p_df["score"]>=72, "✅ STRONG HOLD", "⚖️ HOLD")))

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

def eval_5gw(df): return round(sum((0.9**i)*solve_xi(df, f"xp_{g}")[1] for i, g in enumerate(gws)), 2)

def solve_tr(all_d, c_ids, bnk, k):
    w_sq = all_d[all_d["id"].isin(c_ids)].copy().reset_index(drop=True)
    if k <= 0: return w_sq, [], [], eval_5gw(w_sq)
    rem_b, sold, bought, orig = float(bnk), [], [], set(c_ids)
    for _ in range(int(k)):
        bg, bo_id, bi_r = 0.0, None, None
        c_set, c_cnt = set(w_sq["id"].values), w_sq["team"].value_counts().to_dict()
        for _, o_r in w_sq.iterrows():
            oid = int(o_r["id"])
            if oid not in orig: continue
            cands = all_d[(all_d["pos"]==o_r["pos"]) & (~all_d["id"].isin(c_set)) & (all_d["price"]<=float(o_r["price"])+rem_b+0.001)].sort_values("disc", ascending=False).head(15)
            for _, i_r in cands.iterrows():
                if int(i_r["team"]) != int(o_r["team"]) and c_cnt.get(int(i_r["team"]), 0) >= 3: continue
                gn = float(i_r["disc"]) - float(o_r["disc"])
                if gn > bg: bg, bo_id, bi_r = gn, oid, i_r
        if bo_id is not None and bi_r is not None:
            o_row = w_sq[w_sq["id"]==bo_id].iloc[0]
            sold.append(o_row["web_name"]); bought.append(bi_r["web_name"])
            rem_b += float(o_row["price"]) - float(bi_r["price"])
            w_sq = pd.concat([w_sq[w_sq["id"]!=bo_id], pd.DataFrame([bi_r])], ignore_index=True)
    return w_sq, sold, bought, eval_5gw(w_sq)

auto_ft, auto_b, sq_ids, title = 1, 0.5, [], "Sample Squad (Tap '>' top-left to enter your Manager ID)"
if mid_in.strip().isdigit():
    md = get_mgr(int(mid_in.strip()), nxt - 1)
    if md and "picks" in md["picks"]:
        sq_ids = [p["element"] for p in md["picks"]["picks"]]
        auto_ft = calc_fts(md["hist"])
        auto_b = md["picks"].get("entry_history", {}).get("bank", 0) / 10.0
        title = f"{md['info'].get('name', 'My Team')} (Banked FTs: {auto_ft})"
if not sq_ids:
    for pos, cnt in [("GKP",2), ("DEF",5), ("MID",5), ("FWD",3)]:
        sq_ids.extend(p_df[p_df["pos"]==pos].sort_values("selected_by_percent", ascending=False).head(cnt)["id"].tolist())

with st.sidebar:
    fts = st.slider("Free Transfers (1-5)", 1, 5, int(auto_ft))
    bank = st.number_input("Bank (£m)", 0.0, 25.0, float(auto_b), 0.1)

st.title("⚽ Project Top 10k: FPL Command Centre")
st.subheader(f"📋 {title}")
curr_sq = p_df[p_df["id"].isin(sq_ids)].copy()
t1, t2, t3 = st.tabs([f"🚀 GW{nxt} Roll & Chip Planner", "🪓 Chopping Block", "🔍 Target Hunter"])

with t1:
    _, _, _, xp_0 = solve_tr(p_df, sq_ids, bank, 0)
    sq_1, s1, b1, xp_1 = solve_tr(p_df, sq_ids, bank, 1)
    sq_m, sm, bm, xp_m = solve_tr(p_df, sq_ids, bank, max(2, fts))
    g1, gm, nft = round(xp_1 - xp_0, 2), round(xp_m - xp_0, 2), min(5, fts + 1)
    if fts == 5: st.warning(f"🔥 **AT 5 FT CAP — USE TRANSFER!** Best 1-FT move gains **+{g1} xP**.")
    elif g1 < 2.6: st.info(f"🏦 **VERDICT: ROLL TRANSFER ({fts} → {nft} FTs)** | 1 move gains +{g1} xP; {max(2,fts)}-FT swing gains **+{gm} xP**.")
    else: st.success(f"✅ **VERDICT: USE TRANSFER NOW (+{g1} xP Gain)**")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Roll (Bank {nft} FT)", f"{xp_0} xP", "0 Transfers")
    c2.metric(f"1 FT: {', '.join(s1) or 'Hold'} → {', '.join(b1) or 'Hold'}", f"{xp_1} xP", f"+{g1} xP")
    c3.metric(f"{max(2,fts)}-FT: {', '.join(sm) or 'Hold'} → {', '.join(bm) or 'Hold'}", f"{xp_m} xP", f"+{gm} xP")
    vw = st.radio("View Lineup:", ["Current (Roll)", "Best 1-FT", f"Best {max(2,fts)}-FT"], horizontal=True)
    act = sq_1 if "1-FT" in vw else (sq_m if "-FT" in vw else curr_sq)
    ldf, sxp, bxp = solve_xi(act, f"xp_{nxt}")
    st.markdown(f"#### 🏆 Optimal GW{nxt} XI (**{sxp} xP** | Bench Boost: **+{bxp} xP**)")
    st.dataframe(ldf.sort_values(["start","element_type"], ascending=[False,True])[["web_name","pos","club","price","start","cap",f"op_{nxt}",f"xp_{nxt}","xp_5gw","role"]], use_container_width=True, hide_index=True)

with t2:
    st.dataframe(curr_sq.sort_values("score")[["web_name","pos","club","price","score","verdict","expected_goal_involvements_per_90","defcon_90","xp_5gw","fix_5gw"]], use_container_width=True, hide_index=True)

with t3:
    st.dataframe(p_df[p_df["minutes"]>=180].sort_values("xp_5gw", ascending=False).head(50)[["web_name","pos","club","price","selected_by_percent","role","score","xp_5gw","expected_goal_involvements_per_90","defcon_90","fix_5gw"]], use_container_width=True, hide_index=True)
# --- END OF APP ---