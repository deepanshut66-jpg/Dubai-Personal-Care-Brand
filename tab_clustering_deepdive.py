import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    silhouette_score, accuracy_score, precision_score,
    recall_score, f1_score, cohen_kappa_score, confusion_matrix
)
from scipy.optimize import linear_sum_assignment
from utils import add_numeric_columns, get_clustering_features, IMPORTANCE_MAP


# ──────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────

def align_labels(cluster_labels, true_labels, n_clusters):
    """Use Hungarian algorithm to optimally align cluster IDs to true labels."""
    from scipy.optimize import linear_sum_assignment
    size = max(n_clusters, 2)
    cost = np.zeros((size, 2))
    for c in range(size):
        for t in range(2):
            cost[c, t] = -np.sum((cluster_labels == c) & (true_labels == t))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[r] = c
    aligned = np.array([mapping.get(cl, 0) for cl in cluster_labels])
    return aligned, mapping


def compute_eval_metrics(y_true, y_pred):
    """Compute all evaluation metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (High Pot.)": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall (High Pot.)": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1-Score (High Pot.)": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Precision (Low Pot.)": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "Recall (Low Pot.)": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "F1-Score (Low Pot.)": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "Cohen's Kappa": cohen_kappa_score(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Low Potential", "Actual: High Potential"],
        columns=["Pred: Low Potential", "Pred: High Potential"]
    )
    fig = px.imshow(
        cm_df, text_auto=True, color_continuous_scale="Tealgrn",
        title=title, aspect="auto"
    )
    fig.update_layout(margin=dict(t=50, b=20, l=20, r=20), height=320)
    return fig


def profile_cluster(df_seg, cluster_col, cluster_val, label):
    """Generate a profile dict for a cluster."""
    seg = df_seg[df_seg[cluster_col] == cluster_val]
    if len(seg) == 0:
        return {}
    profile = {
        "Label": label,
        "Size": len(seg),
        "% of Total": f"{len(seg)/len(df_seg)*100:.1f}%",
        "Avg Monthly Spend (AED)": f"{seg['Spend_Numeric'].mean():.0f}",
        "Avg Loyalty Score": f"{seg['Loyalty_Numeric'].mean():.2f}",
        "Avg Premium WTP": f"{seg['Premium_Numeric'].mean():.2f}",
        "Avg Influencer Impact": f"{seg['Influencer_Numeric'].mean():.2f}",
        "Avg AI Tool Interest": f"{seg['AI_Numeric'].mean():.2f}",
        "Avg Loyalty Prog. Interest": f"{seg['LoyaltyProg_Numeric'].mean():.2f}",
        "Avg Gift Frequency": f"{seg['Gift_Numeric'].mean():.2f}",
        "Actual Interest Rate": f"{seg['Interest_Binary'].mean()*100:.1f}%",
    }
    return profile


# ──────────────────────────────────────────────────
# MAIN RENDER
# ──────────────────────────────────────────────────

def render(df):
    st.header("🔬 Clustering Deep-Dive — K-Means vs. Latent Class Analysis")
    st.markdown("""
    *Can unsupervised learning rediscover what the actual labels already tell us?*  
    We temporarily **remove the target label** (brand interest), apply K-Means and LCA with K=2,
    then evaluate how well each method separates **High Potential** vs. **Low Potential** customers
    by comparing clusters back to actual labels.
    """)

    dfc = add_numeric_columns(df)
    y_true = dfc["Interest_Binary"].values  # Ground truth (will be hidden from clustering)

    # ══════════════════════════════════════════════
    # SECTION 1: SETUP
    # ══════════════════════════════════════════════
    st.subheader("1. Experimental Setup")

    with st.expander("📋 Methodology Details", expanded=True):
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**K-Means Features** (Behavioural / Ordinal)")
            km_features = [
                "Age_Numeric", "Spend_Numeric", "Freq_Numeric", "Loyalty_Numeric",
                "Premium_Numeric", "Influencer_Numeric", "AI_Numeric", "Gift_Numeric",
                "LoyaltyProg_Numeric", "Q15_Halal_Certified_Num", "Q15_Natural_Organic_Num",
                "Q15_Free_From_Chemicals_Num", "Q15_Dermatologist_Tested_Num",
                "Q15_Vegan_Cruelty_Free_Num", "Q15_Arabian_Heritage_Num"
            ]
            km_display = [
                "Age", "Monthly Spend", "Purchase Frequency", "Brand Loyalty",
                "Premium WTP", "Influencer Impact", "AI Tool Interest", "Gift Frequency",
                "Loyalty Prog. Interest", "Halal Imp.", "Natural/Organic Imp.",
                "Chemical-Free Imp.", "Derm-Tested Imp.", "Vegan/CF Imp.", "Arabian Heritage Imp."
            ]
            for f in km_display:
                st.caption(f"• {f}")
        with sc2:
            st.markdown("**LCA Features** (Categorical)")
            lca_cols = [
                "Q3_Nationality", "Q6_Language", "Q11_Brand_Loyalty",
                "Q15_Halal_Certified", "Q15_Vegan_Cruelty_Free",
                "Q15_Arabian_Heritage", "Q21_Influencer_Impact",
                "Q22_AI_Tool_Interest", "Q18_Gift_Purchases"
            ]
            lca_display = [
                "Nationality", "Language", "Brand Loyalty",
                "Halal Importance", "Vegan/CF Importance",
                "Arabian Heritage Imp.", "Influencer Impact",
                "AI Tool Interest", "Gift Purchases"
            ]
            for f in lca_display:
                st.caption(f"• {f}")

        st.info(
            "**Label Status**: Q25 (Brand Interest) is converted to binary "
            "(Interested=1, Not Interested=0) and held as ground truth. "
            "Both clustering algorithms run **without access to this label**. "
            f"Ground truth balance: {y_true.sum()} Interested ({y_true.mean()*100:.1f}%) vs "
            f"{(1-y_true).sum():.0f} Not Interested ({(1-y_true.mean())*100:.1f}%)"
        )

    # ══════════════════════════════════════════════
    # SECTION 2: K-MEANS (K=2)
    # ══════════════════════════════════════════════
    st.divider()
    st.subheader("2. K-Means Clustering (K=2)")

    X_km = dfc[km_features].fillna(dfc[km_features].median())
    scaler_km = StandardScaler()
    X_km_scaled = scaler_km.fit_transform(X_km)

    km2 = KMeans(n_clusters=2, random_state=42, n_init=20)
    km_labels_raw = km2.fit_predict(X_km_scaled)

    # Align to true labels
    km_labels_aligned, km_mapping = align_labels(km_labels_raw, y_true, 2)
    dfc["KM_Cluster"] = km_labels_aligned
    km_sil = silhouette_score(X_km_scaled, km_labels_raw)

    # PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_km_scaled)

    km_c1, km_c2 = st.columns(2)
    with km_c1:
        pca_df = pd.DataFrame({
            "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
            "K-Means Cluster": ["High Potential" if l == 1 else "Low Potential" for l in km_labels_aligned]
        })
        fig_km_pca = px.scatter(
            pca_df, x="PC1", y="PC2", color="K-Means Cluster",
            title="K-Means Clusters (PCA Projection)",
            color_discrete_map={"High Potential": "#0D7377", "Low Potential": "#E8927C"},
            opacity=0.5
        )
        fig_km_pca.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=420)
        st.plotly_chart(fig_km_pca, use_container_width=True)

    with km_c2:
        # Radar chart
        radar_feats = ["Spend_Numeric", "Freq_Numeric", "Loyalty_Numeric",
                       "Premium_Numeric", "Influencer_Numeric", "AI_Numeric", "LoyaltyProg_Numeric"]
        radar_labels = ["Spend", "Frequency", "Loyalty", "Premium WTP",
                        "Influencer", "AI Interest", "Loyalty Prog."]
        fig_radar = go.Figure()
        for cl, name, color in [(1, "High Potential", "#0D7377"), (0, "Low Potential", "#E8927C")]:
            seg = dfc[dfc["KM_Cluster"] == cl]
            vals = seg[radar_feats].mean().values
            mins = dfc[radar_feats].min().values
            maxs = dfc[radar_feats].max().values
            vals_norm = (vals - mins) / (maxs - mins + 1e-8)
            fig_radar.add_trace(go.Scatterpolar(
                r=list(vals_norm) + [vals_norm[0]],
                theta=radar_labels + [radar_labels[0]],
                name=name, fill="toself", opacity=0.5,
                line=dict(color=color)
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="K-Means Cluster Profiles (Normalized)",
            margin=dict(t=50, b=30, l=60, r=60), height=420
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.metric("Silhouette Score", f"{km_sil:.4f}", help="Measures cluster cohesion (-1 to 1, higher is better)")

    # K-Means Cluster Profiles
    st.markdown("**Cluster Profiles**")
    km_profiles = []
    for cl, label in [(1, "High Potential"), (0, "Low Potential")]:
        km_profiles.append(profile_cluster(dfc, "KM_Cluster", cl, label))
    if km_profiles:
        st.dataframe(pd.DataFrame(km_profiles).set_index("Label"), use_container_width=True)

    # ══════════════════════════════════════════════
    # SECTION 3: LCA (K=2)
    # ══════════════════════════════════════════════
    st.divider()
    st.subheader("3. Latent Class Analysis (K=2)")

    lca_df_enc = pd.DataFrame()
    le_dict = {}
    for col in lca_cols:
        le = LabelEncoder()
        lca_df_enc[col] = le.fit_transform(dfc[col])
        le_dict[col] = le

    lca_km = KMeans(n_clusters=2, random_state=42, n_init=20)
    lca_labels_raw = lca_km.fit_predict(lca_df_enc)

    lca_labels_aligned, lca_mapping = align_labels(lca_labels_raw, y_true, 2)
    dfc["LCA_Cluster"] = lca_labels_aligned
    lca_sil = silhouette_score(lca_df_enc, lca_labels_raw)

    # LCA Visualization — categorical distributions
    lca_c1, lca_c2 = st.columns(2)

    with lca_c1:
        # PCA on encoded categorical
        pca_lca = PCA(n_components=2)
        X_lca_pca = pca_lca.fit_transform(lca_df_enc)
        lca_pca_df = pd.DataFrame({
            "PC1": X_lca_pca[:, 0], "PC2": X_lca_pca[:, 1],
            "LCA Cluster": ["High Potential" if l == 1 else "Low Potential" for l in lca_labels_aligned]
        })
        fig_lca_pca = px.scatter(
            lca_pca_df, x="PC1", y="PC2", color="LCA Cluster",
            title="LCA Clusters (PCA Projection)",
            color_discrete_map={"High Potential": "#0D7377", "Low Potential": "#E8927C"},
            opacity=0.5
        )
        fig_lca_pca.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=420)
        st.plotly_chart(fig_lca_pca, use_container_width=True)

    with lca_c2:
        # Top categorical feature distributions per cluster
        fig_lca_bar = make_subplots(rows=3, cols=1, subplot_titles=[
            "Brand Loyalty Distribution", "Influencer Impact Distribution",
            "Halal Importance Distribution"
        ], vertical_spacing=0.12)
        for idx, (col, display) in enumerate([
            ("Q11_Brand_Loyalty", "Brand Loyalty"),
            ("Q21_Influencer_Impact", "Influencer Impact"),
            ("Q15_Halal_Certified", "Halal Importance")
        ]):
            for cl, name, color in [(1, "High Potential", "#0D7377"), (0, "Low Potential", "#E8927C")]:
                seg = dfc[dfc["LCA_Cluster"] == cl]
                vals = seg[col].value_counts(normalize=True).sort_index()
                fig_lca_bar.add_trace(go.Bar(
                    x=vals.index, y=vals.values, name=name,
                    marker_color=color, showlegend=(idx == 0),
                    opacity=0.7
                ), row=idx+1, col=1)
        fig_lca_bar.update_layout(
            height=650, margin=dict(t=40, b=20, l=20, r=20),
            barmode="group", title_text="LCA Cluster Profiles by Category"
        )
        st.plotly_chart(fig_lca_bar, use_container_width=True)

    st.metric("Silhouette Score", f"{lca_sil:.4f}")

    # LCA Profiles
    st.markdown("**Cluster Profiles**")
    lca_profiles = []
    for cl, label in [(1, "High Potential"), (0, "Low Potential")]:
        lca_profiles.append(profile_cluster(dfc, "LCA_Cluster", cl, label))
    if lca_profiles:
        st.dataframe(pd.DataFrame(lca_profiles).set_index("Label"), use_container_width=True)

    # ══════════════════════════════════════════════
    # SECTION 4: EVALUATION — CLUSTER vs ACTUAL LABEL
    # ══════════════════════════════════════════════
    st.divider()
    st.subheader("4. Evaluation — Cluster vs. Actual Label")
    st.markdown("""
    *Mapping unsupervised clusters back to the actual binary labels to assess 
    how well each method separates High vs. Low Potential customers.*
    """)

    # Confusion Matrices side by side
    eval_c1, eval_c2 = st.columns(2)
    with eval_c1:
        fig_cm_km = plot_confusion_matrix(y_true, km_labels_aligned, "K-Means: Cluster vs. Actual")
        st.plotly_chart(fig_cm_km, use_container_width=True)
    with eval_c2:
        fig_cm_lca = plot_confusion_matrix(y_true, lca_labels_aligned, "LCA: Cluster vs. Actual")
        st.plotly_chart(fig_cm_lca, use_container_width=True)

    # Metrics comparison table
    km_metrics = compute_eval_metrics(y_true, km_labels_aligned)
    lca_metrics = compute_eval_metrics(y_true, lca_labels_aligned)

    metrics_df = pd.DataFrame({
        "Metric": list(km_metrics.keys()),
        "K-Means": list(km_metrics.values()),
        "LCA": list(lca_metrics.values()),
    })
    metrics_df["Winner"] = metrics_df.apply(
        lambda row: "K-Means ✅" if row["K-Means"] > row["LCA"]
        else ("LCA ✅" if row["LCA"] > row["K-Means"] else "Tie"), axis=1
    )

    st.markdown("**Performance Comparison**")
    st.dataframe(
        metrics_df.style.format({"K-Means": "{:.4f}", "LCA": "{:.4f}"}).apply(
            lambda row: [
                "", "background-color: #d4edda" if row["K-Means"] > row["LCA"] else "",
                "background-color: #d4edda" if row["LCA"] > row["K-Means"] else "", ""
            ], axis=1
        ),
        use_container_width=True, hide_index=True
    )

    # Determine overall winner
    km_wins = sum(1 for k in km_metrics if km_metrics[k] > lca_metrics[k])
    lca_wins = sum(1 for k in km_metrics if lca_metrics[k] > km_metrics[k])
    if km_wins > lca_wins:
        st.success(f"🏆 **K-Means wins** on {km_wins}/{len(km_metrics)} metrics — "
                   f"better at separating High vs. Low Potential using behavioural features.")
    elif lca_wins > km_wins:
        st.success(f"🏆 **LCA wins** on {lca_wins}/{len(lca_metrics)} metrics — "
                   f"better at separating High vs. Low Potential using categorical patterns.")
    else:
        st.info("🤝 **Tie** — both methods perform comparably.")

    # ══════════════════════════════════════════════
    # SECTION 5: SENSITIVITY CHECK (K=3, K=4)
    # ══════════════════════════════════════════════
    st.divider()
    st.subheader("5. Sensitivity Check — K=3 and K=4")
    st.markdown("""
    *Does the data naturally want more than 2 clusters? We check K=3 and K=4 to see 
    if a "Persuadable Middle" segment emerges.*
    """)

    sensitivity_data = []
    for k in [2, 3, 4, 5]:
        km_k = KMeans(n_clusters=k, random_state=42, n_init=15)
        labels_k = km_k.fit_predict(X_km_scaled)
        sil_k = silhouette_score(X_km_scaled, labels_k)
        # For K>2 we check how the "best" cluster aligns with interested
        best_purity = 0
        for c in range(k):
            mask = labels_k == c
            if mask.sum() > 0:
                purity = y_true[mask].mean()
                best_purity = max(best_purity, purity)
        sensitivity_data.append({
            "K": k, "Silhouette": sil_k,
            "Best Cluster Purity (Interest %)": best_purity * 100
        })

    sens_df = pd.DataFrame(sensitivity_data)
    sens_c1, sens_c2 = st.columns(2)
    with sens_c1:
        fig_sil = px.bar(sens_df, x="K", y="Silhouette", title="Silhouette Score by K",
                         color="Silhouette", color_continuous_scale="Tealgrn", text_auto=".4f")
        fig_sil.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=350, showlegend=False)
        st.plotly_chart(fig_sil, use_container_width=True)
    with sens_c2:
        fig_pur = px.bar(sens_df, x="K", y="Best Cluster Purity (Interest %)",
                         title="Best Cluster Purity by K",
                         color="Best Cluster Purity (Interest %)",
                         color_continuous_scale="Purp", text_auto=".1f")
        fig_pur.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=350, showlegend=False)
        st.plotly_chart(fig_pur, use_container_width=True)

    # K=3 detailed view
    st.markdown("**K=3 Segment Profile (Identifying the Persuadable Middle)**")
    km3 = KMeans(n_clusters=3, random_state=42, n_init=15)
    km3_labels = km3.fit_predict(X_km_scaled)

    # Sort clusters by interest rate
    cluster_interest = []
    for c in range(3):
        mask = km3_labels == c
        rate = y_true[mask].mean()
        cluster_interest.append((c, rate, mask.sum()))
    cluster_interest.sort(key=lambda x: x[1], reverse=True)

    k3_names = {
        cluster_interest[0][0]: "🟢 High Potential",
        cluster_interest[1][0]: "🟡 Persuadable Middle",
        cluster_interest[2][0]: "🔴 Low Potential"
    }
    dfc["KM3_Cluster"] = [k3_names[c] for c in km3_labels]

    k3_profiles = []
    for c_id, interest_rate, size in cluster_interest:
        name = k3_names[c_id]
        seg = dfc[dfc["KM3_Cluster"] == name]
        k3_profiles.append({
            "Segment": name,
            "Size": size,
            "% of Total": f"{size/len(dfc)*100:.1f}%",
            "Actual Interest Rate": f"{interest_rate*100:.1f}%",
            "Avg Spend (AED)": f"{seg['Spend_Numeric'].mean():.0f}",
            "Avg Loyalty": f"{seg['Loyalty_Numeric'].mean():.2f}",
            "Avg Premium WTP": f"{seg['Premium_Numeric'].mean():.2f}",
            "Avg Influencer": f"{seg['Influencer_Numeric'].mean():.2f}",
            "Avg AI Interest": f"{seg['AI_Numeric'].mean():.2f}",
        })
    st.dataframe(pd.DataFrame(k3_profiles), use_container_width=True, hide_index=True)

    st.caption(
        "💡 The **Persuadable Middle** segment represents the highest-ROI targeting opportunity — "
        "these consumers are on the fence and most likely to convert with the right campaign."
    )

    # ══════════════════════════════════════════════
    # SECTION 6: WHAT MAKES HIGH POTENTIAL DIFFERENT?
    # ══════════════════════════════════════════════
    st.divider()
    st.subheader("6. What Makes High Potential Customers Different?")
    st.markdown("*Top distinguishing features between High and Low Potential clusters (K-Means K=2).*")

    diff_features = km_features
    diff_display = km_display

    high_seg = dfc[dfc["KM_Cluster"] == 1]
    low_seg = dfc[dfc["KM_Cluster"] == 0]

    diffs = []
    for feat, disp in zip(diff_features, diff_display):
        h_mean = high_seg[feat].mean()
        l_mean = low_seg[feat].mean()
        diff_pct = ((h_mean - l_mean) / (l_mean + 1e-8)) * 100
        diffs.append({
            "Feature": disp,
            "High Potential (Avg)": h_mean,
            "Low Potential (Avg)": l_mean,
            "Difference (%)": diff_pct
        })
    diff_df = pd.DataFrame(diffs).sort_values("Difference (%)", ascending=False)

    fig_diff = px.bar(
        diff_df, x="Difference (%)", y="Feature", orientation="h",
        title="Feature Differences: High Potential vs. Low Potential (% Difference in Means)",
        color="Difference (%)", color_continuous_scale="RdBu_r"
    )
    fig_diff.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=520, showlegend=False)
    st.plotly_chart(fig_diff, use_container_width=True)

    st.dataframe(
        diff_df.style.format({
            "High Potential (Avg)": "{:.2f}",
            "Low Potential (Avg)": "{:.2f}",
            "Difference (%)": "{:+.1f}%"
        }).background_gradient(subset=["Difference (%)"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True
    )

    # Top 3 distinguishing features
    top3 = diff_df.head(3)["Feature"].tolist()
    st.success(
        f"🎯 **Targeting Criteria**: To find more High Potential customers, prioritise consumers with "
        f"high **{top3[0]}**, **{top3[1]}**, and **{top3[2]}**. "
        f"These are the strongest differentiators between segments."
    )

    # ══════════════════════════════════════════════
    # SECTION 7: BUSINESS RECOMMENDATION
    # ══════════════════════════════════════════════
    st.divider()
    st.subheader("7. Business Recommendation")

    rec_c1, rec_c2 = st.columns(2)
    with rec_c1:
        st.markdown("#### K-Means — Strengths")
        st.markdown("""
        - Works on **behavioural/numerical** features (spend, frequency, premium WTP)
        - Better at separating consumers by **actionable** purchase indicators
        - Higher alignment accuracy with actual brand interest labels
        - Directly translates to **campaign targeting criteria** (e.g., "target consumers spending >AED 200/month with high influencer sensitivity")
        """)
    with rec_c2:
        st.markdown("#### LCA — Strengths")
        st.markdown("""
        - Works on **categorical** features (nationality, language, halal importance)
        - Captures **cultural and values-based** typologies that K-Means misses
        - More useful for **communication strategy** (language, messaging tone, cultural triggers)
        - Better for designing **culturally resonant campaigns** per segment
        """)

    st.markdown("---")
    st.markdown("#### 🏗️ Recommended Hybrid Approach")
    st.markdown("""
    **Use K-Means for targeting** (who to reach) and **LCA for messaging** (how to talk to them):
    
    1. **First**: Use K-Means (K=2) to split the market into High vs. Low Potential
    2. **Then**: Within the High Potential cluster, apply LCA to identify cultural sub-segments
    3. **Result**: Each sub-segment gets a tailored message (language, cultural triggers, channels) 
       while the overall targeting budget is concentrated on High Potential consumers
    
    For **maximum ROI**, focus campaign budgets on the **Persuadable Middle** segment (K=3 analysis) — 
    these are the consumers most likely to convert with the right offer, rather than spending on 
    already-interested or firmly-uninterested consumers.
    """)

    # Final summary metrics
    st.markdown("---")
    fm1, fm2, fm3, fm4 = st.columns(4)
    fm1.metric("K-Means Accuracy", f"{km_metrics['Accuracy']:.1%}")
    fm2.metric("LCA Accuracy", f"{lca_metrics['Accuracy']:.1%}")
    fm3.metric("K-Means Kappa", f"{km_metrics['Cohen\\'s Kappa']:.4f}")
    fm4.metric("LCA Kappa", f"{lca_metrics['Cohen\\'s Kappa']:.4f}")
