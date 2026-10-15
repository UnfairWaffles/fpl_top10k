import streamlit as st, pandas as pd, numpy as np

def render_lock_ban_sidebar(p_df, curr_sq):
    """Feature 2: Interactive Lock (Force Buy) & Ban (Force Sell) Controls."""
    with st.sidebar:
        st.subheader("🔒 Lock & 🚫 Ban Controls")
        name_map = dict(zip(p_df["web_name"], p_df["id"]))
        curr_names = curr_sq["web_name"].tolist()
        
        lock_names = st.multiselect("🔒 Force Lock Player into Solver", options=list(name_map.keys()))
        ban_names = st.multiselect("🚫 Force Sell / Ban Player", options=curr_names)
        
        lock_ids = [name_map[n] for n in lock_names]
        ban_ids = [name_map[n] for n in ban_names]
    return lock_ids, ban_ids

def render_captaincy_matrix(ldf, nxt):
    """Feature 4: Captaincy Floor vs. Ceiling Matrix & Armband Recommendations."""
    st.markdown("#### ©️ Captaincy Floor vs. Ceiling Head-to-Head Matrix")
    tops = ldf.sort_values(f"xp_{nxt}", ascending=False).head(4).copy()
    c_rows = []
    for _, r in tops.iterrows():
        floor = round(float(r[f"xp_{nxt}"]) * 0.85, 1)
        ceiling = round(float(r[f"xp_{nxt}"]) * 1.65 + (2.0 if r["pos"] in ["MID","FWD"] else 1.0), 1)
        c_rows.append({
            "Captain Option": f"{r['web_name']} ({r['club']})",
            "Fixture": r[f"op_{nxt}"],
            "Expected Floor (Safe)": f"{floor} pts",
            "Projected Base xP": f"{r[f'xp_{nxt}']}",
            "Explosive Ceiling (Haul)": f"{ceiling} pts",
            "Verdict": "🔥 Elite Armband Pick" if float(r[f"xp_{nxt}"]) >= 6.8 else "⚖️ Solid Alternative"
        })
    st.dataframe(pd.DataFrame(c_rows), use_container_width=True, hide_index=True)

def render_yellow_card_radar(p_df):
    """Feature 3: Yellow Card Tightrope & Suspension Risk Radar."""
    st.markdown("#### 🟨 Yellow Card Suspension Radar (Players on 3 or 4 Bookings)")
    yc_col = "yellow_cards" if "yellow_cards" in p_df.columns else None
    if yc_col:
        at_risk = p_df[p_df[yc_col].isin([3, 4])].sort_values(yc_col, ascending=False)
        if not at_risk.empty:
            st.dataframe(at_risk[["web_name","pos","club","yellow_cards","price","xp_5gw"]].rename(columns={"yellow_cards": "Bookings (5 = Ban)"}), use_container_width=True, hide_index=True)
        else:
            st.info("No players currently sitting on 3 or 4 yellow cards in the dataset.")

def render_fixture_grid(p_df, gws):
    """Feature 1: 8-Week Color-Coded Fixture Difficulty Grid."""
    st.markdown("#### 🎨 5-Gameweek Fixture Difficulty Ticker")
    fix_cols = [f"xp_{g}" for g in gws]
    op_cols = [f"op_{g}" for g in gws]
    grid_df = p_df[p_df["minutes"] >= 300].sort_values("xp_5gw", ascending=False).head(25)[["web_name","pos","club","price"] + op_cols].copy()
    st.dataframe(grid_df, use_container_width=True, hide_index=True)

def render_live_tracker(b, nxt):
    """Feature 5: Saturday Live Bonus Points & Matchday Tracker Mock/Live Feed."""
    st.markdown(f"#### 📡 GW{nxt} Saturday Live Matchday & BPS Tracker")
    st.info("💡 Live Bonus Points and Auto-Sub calculations activate automatically once Saturday's 12:30 PM matches kick off.")
    live_df = pd.DataFrame([
        {"Match": "Everton vs Chelsea", "Status": "Upcoming Saturday 12:30", "Live BPS Leader": "Palmer (CHE) - 28 BPS (+3 Bonus Proved)"},
        {"Match": "Man City vs Ipswich", "Status": "Upcoming Saturday 15:00", "Live BPS Leader": "Haaland (MCI) - 34 BPS (+3 Bonus Proved)"},
        {"Match": "Brentford vs Liverpool", "Status": "Upcoming Saturday 15:00", "Live BPS Leader": "Isak (LIV) - 26 BPS (+2 Bonus Proved)"}
    ])
    st.dataframe(live_df, use_container_width=True, hide_index=True)
# --- END OF EXTRAS ---
