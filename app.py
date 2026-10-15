import streamlit as st, pandas as pd, numpy as np
import engine as eg
import extras as ex

st.set_page_config(page_title="Project Top 10k Pro", page_icon="⚽", layout="wide")

b, f = eg.load_fpl()
nxt = int(next((e["id"] for e in b["events"] if e.get("is_next")), b["events"][-1]["id"]))

with st.sidebar:
    st.header("⚙️ Team & League Setup")
    mid_in = st.text_input("Your FPL Manager ID", "", placeholder="e.g. 1234567")
    lid_in = st.text_input("Mini-League ID (Optional)", "", placeholder="e.g. 314159")
    mode = st.selectbox("Strategy Mode", ["Balanced", "Sword (<10% Diffs)", "Shield (High EO)"])
    fix_w = st.slider("Fixture Weighting", 0.10, 0.85, 0.45, 0.05)

p_df, gws = eg.build_df(b, f, nxt, fix_w, mode)

auto_ft, auto_b, sq_ids, tr_list = 1, 0.5, [], []
title = "Sample Squad (Tap '>' top-left to enter your Manager ID)"
if mid_in.strip().isdigit():
    md = eg.get_mgr(int(mid_in.strip()), nxt - 1)
    if md and "picks" in md["picks"]:
        sq_ids = [p["element"] for p in md["picks"]["picks"]]
        auto_ft = eg.calc_fts(md["hist"])
        auto_b = md["picks"].get("entry_history", {}).get("bank", 0) / 10.0
        tr_list = md.get("tr", [])
        title = f"{md['info'].get('name', 'My Team')} (Banked FTs: {auto_ft})"

if not sq_ids:
    for pos, cnt in [("GKP",2), ("DEF",5), ("MID",5), ("FWD",3)]:
        sq_ids.extend(p_df[p_df["pos"]==pos].sort_values("selected_by_percent", ascending=False).head(cnt)["id"].tolist())

p_df = eg.apply_selling_prices(p_df, sq_ids, tr_list)
curr_sq = p_df[p_df["id"].isin(sq_ids)].copy().reset_index(drop=True)

# Lock & Ban sidebar controls
lock_ids, ban_ids = ex.render_lock_ban_sidebar(p_df, curr_sq)

with st.sidebar:
    fts = st.slider("Free Transfers (1-5)", 1, 5, int(auto_ft))
    bank = st.number_input("Money in Bank (£m)", 0.0, 25.0, float(auto_b), 0.1)
    tot_sell_bud = round(float(curr_sq["sell_price"].sum()) + bank, 1)
    st.caption(f"💰 Total Realizable Value: **£{tot_sell_bud}m**")

st.title("⚽ Project Top 10k: Pro Command Centre")
st.subheader(f"📋 {title}")

t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    f"🚀 GW{nxt} Planner & Combos",
    "🪓 Squad & Selling Prices",
    "🔥 Target & Momentum",
    "🃏 Wildcard & Free Hit",
    "💷 Price Radar & YC",
    "🕵️ Mini-League Spy",
    "📡 Matchday Live"
])

with t1:
    xp_0 = eg.eval_5gw(curr_sq, gws)
    best_1g, s1, b1, sq_1 = 0.0, [], [], curr_sq.copy()
    c_set, c_cnt = set(sq_ids), curr_sq["team"].value_counts().to_dict()
    
    # Filter banned players out of solver consideration
    valid_p_df = p_df[~p_df["id"].isin(ban_ids)].copy()

    for _, o_r in curr_sq.iterrows():
        if int(o_r["id"]) in ban_ids: continue
        cands = valid_p_df[(valid_p_df["pos"]==o_r["pos"]) & (~valid_p_df["id"].isin(c_set)) & (valid_p_df["price"]<=float(o_r["sell_price"])+bank+0.001)].sort_values("disc", ascending=False).head(15)
        for _, i_r in cands.iterrows():
            if int(i_r["team"]) != int(o_r["team"]) and c_cnt.get(int(i_r["team"]), 0) >= 3: continue
            gn = float(i_r["disc"]) - float(o_r["disc"])
            if gn > best_1g:
                best_1g, s1, b1 = gn, [o_r["web_name"]], [i_r["web_name"]]
                sq_1 = pd.concat([curr_sq[curr_sq["id"]!=int(o_r["id"])], pd.DataFrame([i_r])], ignore_index=True)
    xp_1 = eg.eval_5gw(sq_1, gws)

    sq_2, s2, b2, xp_2 = eg.solve_2ft_combo(valid_p_df, sq_ids, bank, gws)
    g1, g2, nft = round(xp_1 - xp_0, 2), round(xp_2 - xp_0, 2), min(5, fts + 1)

    if fts == 5: st.warning(f"🔥 **AT 5 FT CAP — USE TRANSFER!** Best 1-FT gains **+{g1} xP**; 2-FT Combo gains **+{g2} xP**.")
    elif g1 < 2.6 and (fts == 1 or g2 < 4.5): st.info(f"🏦 **VERDICT: ROLL TRANSFER ({fts} → {nft} FTs)** | Banking unlocks a 2-Player Combo worth **+{g2} xP**.")
    else: st.success(f"✅ **VERDICT: ATTACK THIS WEEK** | 1-FT gains **+{g1} xP** | 2-Player Combo gains **+{g2} xP**.")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Roll (Bank {nft} FT)", f"{xp_0} xP", "0 Moves")
    c2.metric(f"1-FT: {', '.join(s1) or 'Hold'} → {', '.join(b1) or 'Hold'}", f"{xp_1} xP", f"+{g1} xP")
    c3.metric(f"2-FT Combo: {', '.join(s2) or 'Hold'} → {', '.join(b2) or 'Hold'}", f"{xp_2} xP", f"+{g2} xP")

    vw = st.radio("Inspect XI For:", ["Current Squad (Roll)", "After 1-FT Move", "After 2-FT Linked Combo"], horizontal=True)
    act = sq_1 if "1-FT" in vw else (sq_2 if "2-FT" in vw else curr_sq)
    ldf, sxp, bxp = eg.solve_xi(act, f"xp_{nxt}")
    
    # Render Captaincy Matrix
    ex.render_captaincy_matrix(ldf, nxt)

    st.markdown(f"#### 🏆 Optimal GW{nxt} XI (**{sxp} xP** | Bench Boost Value: **+{bxp} xP**)")
    st.dataframe(ldf.sort_values(["start","element_type"], ascending=[False,True])[["web_name","pos","club","price","sell_price","start","cap",f"op_{nxt}",f"xp_{nxt}","xp_5gw","surge_tag","role"]], use_container_width=True, hide_index=True)

with t2:
    st.markdown("#### 🪓 Squad Audit & Locked-In Value (`sell_price` vs current `price`)")
    st.dataframe(curr_sq.sort_values("score")[["web_name","pos","club","price","sell_price","locked_val","score","verdict","momentum","defcon_90","xp_5gw","fix_5gw"]], use_container_width=True, hide_index=True)

with t3:
    f_pos = st.multiselect("Filter Position", ["GKP","DEF","MID","FWD"], default=["DEF","MID","FWD"])
    st.dataframe(p_df[(p_df["minutes"]>=180) & (p_df["pos"].isin(f_pos))].sort_values("xp_5gw", ascending=False).head(50)[["web_name","pos","club","price","selected_by_percent","role","surge_tag","momentum","score","xp_5gw","expected_goal_involvements_per_90","defcon_90","fix_5gw"]], use_container_width=True, hide_index=True)

with t4:
    st.markdown(f"#### 🃏 Chip Simulator (Budget: **£{tot_sell_bud}m** | First-Half Chips Expire GW19)")
    wc_df, wc_cost, wc_5gw = eg.build_wildcard(p_df, tot_sell_bud, "disc", gws)
    fh_df, fh_cost, _ = eg.build_wildcard(p_df, tot_sell_bud, f"xp_{nxt}", gws)
    _, fh_gw1, _ = eg.solve_xi(fh_df, f"xp_{nxt}")
    _, cur_gw1, _ = eg.solve_xi(curr_sq, f"xp_{nxt}")
    wc_gain = round(wc_5gw - xp_0, 2)
    fh_gain = round(fh_gw1 - cur_gw1, 2)
    k1, k2 = st.columns(2)
    k1.metric("5-Week Wildcard Squad xP", f"{wc_5gw} xP (£{wc_cost}m)", f"+{wc_gain} xP vs Current Squad ({'🔥 PLAY WC' if wc_gain>=22 else '🏦 HOLD WC'})")
    k2.metric(f"GW{nxt} Free Hit XI xP", f"{fh_gw1} xP (£{fh_cost}m)", f"+{fh_gain} xP this week ({'🔥 PLAY FH' if fh_gain>=14 else '🏦 HOLD FH'})")
    chip_v = st.radio("View Optimal Draft:", ["5-Week Wildcard Squad", f"GW{nxt} Single-Week Free Hit"], horizontal=True)
    c_show, _, _ = eg.solve_xi(wc_df if "Wildcard" in chip_v else fh_df, f"xp_{nxt}")
    st.dataframe(c_show.sort_values(["start","element_type"], ascending=[False,True])[["web_name","pos","club","price","start","cap",f"op_{nxt}",f"xp_{nxt}","xp_5gw","fix_5gw"]], use_container_width=True, hide_index=True)

with t5:
    st.markdown("#### 💷 Tonight's Transfer Velocity Radar & Suspension Risk")
    ex.render_yellow_card_radar(p_df)
    st.divider()
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("##### 📈 Top Risers (Buy Early)")
        st.dataframe(p_df.sort_values("net_tr", ascending=False).head(15)[["web_name","pos","club","price","net_tr","price_trend","xp_5gw"]], use_container_width=True, hide_index=True)
    with p2:
        st.markdown("##### 📉 Top Fallers (Sell Warning)")
        st.dataframe(p_df.sort_values("net_tr", ascending=True).head(15)[["web_name","pos","club","price","net_tr","price_trend","xp_5gw"]], use_container_width=True, hide_index=True)

with t6:
    if not lid_in.strip().isdigit():
        st.info("👈 Enter your **Mini-League ID** in the sidebar to spy on your top 5 rivals! (Find the number in the URL when viewing your mini-league table on the FPL website).")
    else:
        lg = eg.get_league(int(lid_in.strip()), nxt - 1)
        if not lg or not lg["rivals"]:
            st.error("Could not load that Mini-League ID. Make sure it's a Classic League ID.")
        else:
            st.markdown(f"#### 🕵️ {lg['name']} — Top 5 Rivals Audit")
            id_to_name = dict(zip(p_df["id"], p_df["web_name"]))
            rival_rows, all_rival_picks = [], []
            for rv in lg["rivals"]:
                all_rival_picks.extend(rv["ids"])
                diff_ids = [x for x in rv["ids"] if x not in sq_ids]
                rival_rows.append({"Team": rv["name"], "Manager": rv["mgr"], "Points": rv["pts"], "Last Captain": id_to_name.get(rv["cap"], "-"), "Players They Have That You Don't": ", ".join(id_to_name.get(x, "") for x in diff_ids[:6])})
            st.dataframe(pd.DataFrame(rival_rows), use_container_width=True, hide_index=True)
            counts = pd.Series(all_rival_picks).value_counts()
            danger_ids = [pid for pid, cnt in counts.items() if pid not in sq_ids and cnt >= 2]
            sword_ids = [pid for pid in p_df.sort_values("xp_5gw", ascending=False).head(30)["id"] if pid not in set(all_rival_picks)]
            r1, r2 = st.columns(2)
            with r1:
                st.markdown("##### ⚠️ Rival Blockers (Owned by 2+ Rivals, Not You)")
                st.dataframe(p_df[p_df["id"].isin(danger_ids)][["web_name","pos","club","price",f"xp_{nxt}","xp_5gw"]], use_container_width=True, hide_index=True)
            with r2:
                st.markdown("##### ⚔️ Pure Mini-League Differentials (0% Rival Owned)")
                st.dataframe(p_df[p_df["id"].isin(sword_ids)].head(10)[["web_name","pos","club","price",f"xp_{nxt}","xp_5gw","fix_5gw"]], use_container_width=True, hide_index=True)

with t7:
    ex.render_live_tracker(b, nxt)
# --- END OF APP ---
