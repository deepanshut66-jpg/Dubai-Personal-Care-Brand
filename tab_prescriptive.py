import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import add_numeric_columns, get_clustering_features, explode_multiselect, SPEND_MAP


PRODUCT_LINES = {
    "Desert Shield": "Climate-adaptive SPF + hydration skincare (unisex)",
    "Oud & Element": "Heritage-meets-modern men's grooming (oud-infused)",
    "Glow Ritual": "K-beauty-inspired skincare for melanin-rich skin",
    "Saha": "Halal-certified natural daily essentials (mass market)",
    "Luma": "AI-personalised D2C skincare subscription",
    "Ramadan Radiance": "Seasonal limited-edition gift collections"
}


def render(df):
    st.header("💡 Prescriptive Analytics — What Should We Do?")
    st.markdown("*Translating data insights into segment-specific go-to-market strategies.*")

    dfc = add_numeric_columns(df)

    if "kmeans_model" not in st.session_state:
        st.warning("⚠️ Please run the **Predictive Analytics** tab first to generate clusters.")
        st.stop()

    km = st.session_state["kmeans_model"]
    scaler = st.session_state["kmeans_scaler"]
    feat_names = st.session_state["kmeans_features"]
    cluster_names = st.session_state["cluster_names"]

    X_scaled, _, _ = get_clustering_features(dfc)
    dfc["Cluster"] = km.predict(X_scaled)
    dfc["Cluster_Name"] = dfc["Cluster"].map(cluster_names)

    st.subheader("Segment Strategy Dashboard")

    for c in sorted(cluster_names.keys()):
        seg = dfc[dfc["Cluster"] == c]
        seg_name = cluster_names[c]
        pct = len(seg) / len(dfc) * 100

        with st.expander(f"🎯 {seg_name} — {len(seg):,} consumers ({pct:.1f}%)", expanded=(c == 0)):

            # Key Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Monthly Spend", f"AED {seg['Spend_Numeric'].mean():.0f}")
            m2.metric("Avg Loyalty Score", f"{seg['Loyalty_Numeric'].mean():.1f}/5")
            m3.metric("Avg Premium WTP", f"{seg['Premium_Numeric'].mean():.1f}/4")
            interest_rate = seg["Interest_Binary"].mean() * 100
            m4.metric("Brand Interest Rate", f"{interest_rate:.0f}%")

            sc1, sc2 = st.columns(2)

            # Demographics
            with sc1:
                st.markdown("**👤 Demographics**")
                top_nat = seg["Q3_Nationality"].value_counts().head(3)
                for nat, cnt in top_nat.items():
                    short = nat.split("(")[0].strip() if "(" in nat else nat
                    st.caption(f"• {short}: {cnt/len(seg)*100:.0f}%")
                top_age = seg["Q1_Age_Group"].mode()[0]
                st.caption(f"• Dominant Age: {top_age}")
                top_inc = seg["Q5_Income"].value_counts().head(2)
                for inc, cnt in top_inc.items():
                    st.caption(f"• Income {inc}: {cnt/len(seg)*100:.0f}%")

            # Channel & Discovery
            with sc2:
                st.markdown("**🛒 Channel Strategy**")
                ch_counts = explode_multiselect(seg, "Q9_Channels").head(3)
                for ch, cnt in ch_counts.items():
                    short_ch = ch.split("(")[0].strip() if "(" in ch else ch
                    st.caption(f"• {short_ch}: {cnt/len(seg)*100:.0f}%")
                st.markdown("**📱 Discovery**")
                disc_counts = explode_multiselect(seg, "Q20_Discovery_Methods").head(3)
                for d, cnt in disc_counts.items():
                    st.caption(f"• {d}: {cnt/len(seg)*100:.0f}%")

            sc3, sc4 = st.columns(2)

            # Product Recommendation
            with sc3:
                st.markdown("**📦 Recommended Product Lines**")
                avg_spend = seg["Spend_Numeric"].mean()
                avg_premium = seg["Premium_Numeric"].mean()
                avg_halal = seg["Q15_Halal_Certified_Num"].mean() if "Q15_Halal_Certified_Num" in seg else 2
                avg_heritage = seg["Q15_Arabian_Heritage_Num"].mean() if "Q15_Arabian_Heritage_Num" in seg else 2
                avg_ai = seg["AI_Numeric"].mean()

                recommendations = []
                if avg_spend >= 250 and avg_premium >= 2.5:
                    recommendations.append("Oud & Element")
                    recommendations.append("Luma")
                if avg_halal >= 2.8 or avg_heritage >= 2.5:
                    recommendations.append("Saha")
                    recommendations.append("Ramadan Radiance")
                if avg_spend >= 150:
                    recommendations.append("Desert Shield")
                    recommendations.append("Glow Ritual")
                if avg_ai >= 3:
                    recommendations.append("Luma")
                if len(recommendations) == 0:
                    recommendations = ["Saha", "Desert Shield"]
                recommendations = list(dict.fromkeys(recommendations))[:3]

                for rec in recommendations:
                    st.caption(f"✅ **{rec}** — {PRODUCT_LINES[rec]}")

            # Pricing & Promotion
            with sc4:
                st.markdown("**💰 Pricing & Promotion**")
                top_disc = seg["Q23_Discount_Type"].value_counts().head(2)
                for d, cnt in top_disc.items():
                    short_d = d.split("(")[0].strip() if "(" in d else d
                    st.caption(f"• {short_d}: {cnt/len(seg)*100:.0f}%")
                avg_lp = seg["LoyaltyProg_Numeric"].mean()
                lp_label = "High" if avg_lp >= 3.5 else "Medium" if avg_lp >= 2.5 else "Low"
                st.caption(f"• Loyalty Programme Affinity: **{lp_label}** ({avg_lp:.1f}/5)")

                st.markdown("**📅 Seasonal Triggers**")
                fest_counts = explode_multiselect(seg, "Q17_Festival_Triggers").head(3)
                for f, cnt in fest_counts.items():
                    st.caption(f"• {f}: {cnt/len(seg)*100:.0f}%")

            # Communication
            st.markdown("**🗣️ Communication Strategy**")
            lang_dist = seg["Q6_Language"].value_counts().head(3)
            lang_str = " | ".join([f"{l}: {c/len(seg)*100:.0f}%" for l, c in lang_dist.items()])
            st.caption(f"Languages: {lang_str}")
            avg_infl = seg["Influencer_Numeric"].mean()
            infl_label = "High" if avg_infl >= 3 else "Medium" if avg_infl >= 2 else "Low"
            st.caption(f"Influencer Sensitivity: **{infl_label}** ({avg_infl:.1f}/4)")

    # Overall comparison
    st.divider()
    st.subheader("Segment Comparison Matrix")
    comparison_data = []
    for c in sorted(cluster_names.keys()):
        seg = dfc[dfc["Cluster"] == c]
        comparison_data.append({
            "Segment": cluster_names[c],
            "Size (%)": f"{len(seg)/len(dfc)*100:.1f}%",
            "Avg Spend (AED)": f"{seg['Spend_Numeric'].mean():.0f}",
            "Brand Interest": f"{seg['Interest_Binary'].mean()*100:.0f}%",
            "Loyalty": f"{seg['Loyalty_Numeric'].mean():.1f}",
            "Premium WTP": f"{seg['Premium_Numeric'].mean():.1f}",
            "Influencer": f"{seg['Influencer_Numeric'].mean():.1f}",
        })
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
