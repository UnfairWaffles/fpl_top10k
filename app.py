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
p_df["verdict"] = np.where(p_df
