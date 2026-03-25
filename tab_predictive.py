import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
from utils import add_numeric_columns, get_classification_features, get_clustering_features


def render(df):
    st.header("🤖 Predictive Analytics — What Will Happen?")
    st.markdown("*Machine learning models to predict brand interest and discover consumer segments.*")

    dfc = add_numeric_columns(df)

    pred_tab1, pred_tab2, pred_tab3 = st.tabs([
        "🎯 Classification", "🧩 K-Means Clustering", "🔬 Latent Class Analysis"
    ])

    # ==================== CLASSIFICATION ====================
    with pred_tab1:
        st.subheader("Classification — Predicting Brand Interest")
        st.markdown("""
        **Target**: Q25 Brand Interest → Binary (Interested vs Not Interested)  
        **Models**: Decision Tree, Random Forest, Gradient Boosted Tree
        """)

        X, y, feature_cols, encoders = get_classification_features(dfc)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        models = {
            "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
            "Gradient Boosted Tree": GradientBoostingClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
            )
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            results[name] = {
                "model": model,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "auc": roc_auc_score(y_test, y_prob)
            }

        # Metrics table
        st.subheader("Model Performance Comparison")
        metrics_df = pd.DataFrame({
            "Model": list(results.keys()),
            "Accuracy": [r["accuracy"] for r in results.values()],
            "Precision": [r["precision"] for r in results.values()],
            "Recall": [r["recall"] for r in results.values()],
            "F1-Score": [r["f1"] for r in results.values()],
            "AUC-ROC": [r["auc"] for r in results.values()]
        })
        st.dataframe(
            metrics_df.style.format({
                "Accuracy": "{:.4f}", "Precision": "{:.4f}",
                "Recall": "{:.4f}", "F1-Score": "{:.4f}", "AUC-ROC": "{:.4f}"
            }).highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
                            color="#0D737730"),
            use_container_width=True
        )

        best_name = max(results, key=lambda k: results[k]["auc"])
        st.success(f"🏆 Best Model: **{best_name}** (AUC-ROC: {results[best_name]['auc']:.4f})")

        # ROC Curves
        st.subheader("ROC Curves")
        fig_roc = go.Figure()
        colors = ["#0D7377", "#E8927C", "#6B4C9A"]
        for (name, r), color in zip(results.items(), colors):
            fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{name} (AUC={r['auc']:.4f})",
                line=dict(color=color, width=2)
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random", line=dict(color="gray", dash="dash")
        ))
        fig_roc.update_layout(
            title="ROC Curves — All Models",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            margin=dict(t=50, b=30, l=30, r=30), height=450
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        # Confusion Matrices
        st.subheader("Confusion Matrices")
        cm_cols = st.columns(3)
        for idx, (name, r) in enumerate(results.items()):
            cm = confusion_matrix(y_test, r["y_pred"])
            cm_df = pd.DataFrame(cm,
                                 index=["Actual: Not Interested", "Actual: Interested"],
                                 columns=["Pred: Not Interested", "Pred: Interested"])
            with cm_cols[idx]:
                st.markdown(f"**{name}**")
                fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Tealgrn",
                                   aspect="auto")
                fig_cm.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=280)
                st.plotly_chart(fig_cm, use_container_width=True)

        # Feature Importance
        st.subheader("Feature Importance (Best Model)")
        best_model = results[best_name]["model"]
        importance = best_model.feature_importances_
        feat_imp_df = pd.DataFrame({
            "Feature": feature_cols, "Importance": importance
        }).sort_values("Importance", ascending=True)
        fig_imp = px.bar(feat_imp_df, x="Importance", y="Feature", orientation="h",
                         title=f"Feature Importance — {best_name}",
                         color="Importance", color_continuous_scale="Tealgrn")
        fig_imp.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=550, showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)

        # Store best model in session
        st.session_state["best_model"] = best_model
        st.session_state["best_model_name"] = best_name
        st.session_state["feature_cols"] = feature_cols
        st.session_state["encoders"] = encoders

    # ==================== K-MEANS ====================
    with pred_tab2:
        st.subheader("K-Means Clustering — Behavioural Segmentation")
        st.markdown("*Grouping consumers based on behavioural similarity to discover natural market segments.*")

        X_scaled, feat_names, scaler = get_clustering_features(dfc)

        # Elbow + Silhouette
        from sklearn.metrics import silhouette_score
        K_range = range(2, 9)
        inertias = []
        sil_scores = []
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X_scaled, km.labels_, sample_size=2000))

        ec1, ec2 = st.columns(2)
        with ec1:
            fig_elbow = px.line(x=list(K_range), y=inertias, markers=True,
                                title="Elbow Method", labels={"x": "K", "y": "Inertia"})
            fig_elbow.update_traces(line_color="#0D7377")
            fig_elbow.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=350)
            st.plotly_chart(fig_elbow, use_container_width=True)
        with ec2:
            fig_sil = px.line(x=list(K_range), y=sil_scores, markers=True,
                              title="Silhouette Scores", labels={"x": "K", "y": "Score"})
            fig_sil.update_traces(line_color="#E8927C")
            fig_sil.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=350)
            st.plotly_chart(fig_sil, use_container_width=True)

        optimal_k = st.slider("Select number of clusters (K):", 2, 8, 4, key="kmeans_k")
        km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        dfc["Cluster"] = km_final.fit_predict(X_scaled)

        # Cluster names based on centroids
        cluster_names = {}
        centroids = pd.DataFrame(km_final.cluster_centers_, columns=feat_names)
        name_templates = [
            "Premium Digital Natives", "Value-Driven Family Shoppers",
            "Heritage-Conscious Loyalists", "Deal-Seeking Explorers",
            "Eco-Conscious Millennials", "Status-Driven Spenders",
            "Practical Minimalists", "Social Influencee Buyers"
        ]
        for c in range(optimal_k):
            cluster_names[c] = name_templates[c] if c < len(name_templates) else f"Cluster {c}"
        dfc["Cluster_Name"] = dfc["Cluster"].map(cluster_names)

        # PCA Scatter
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        dfc["PCA1"] = X_pca[:, 0]
        dfc["PCA2"] = X_pca[:, 1]

        fig_pca = px.scatter(dfc, x="PCA1", y="PCA2", color="Cluster_Name",
                             title="Consumer Segments (PCA Projection)",
                             color_discrete_sequence=px.colors.qualitative.Bold,
                             opacity=0.6)
        fig_pca.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=500)
        st.plotly_chart(fig_pca, use_container_width=True)

        # Cluster size
        cluster_size = dfc["Cluster_Name"].value_counts().reset_index()
        cluster_size.columns = ["Segment", "Count"]
        fig_cs = px.pie(cluster_size, values="Count", names="Segment",
                        title="Cluster Distribution", hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Bold)
        fig_cs.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=400)
        st.plotly_chart(fig_cs, use_container_width=True)

        # Radar chart for centroids
        st.subheader("Cluster Profiles — Radar Chart")
        radar_features = ["Spend_Numeric", "Freq_Numeric", "Loyalty_Numeric",
                          "Premium_Numeric", "Influencer_Numeric", "AI_Numeric",
                          "LoyaltyProg_Numeric"]
        radar_labels = ["Spend", "Frequency", "Loyalty", "Premium WTP",
                        "Influencer", "AI Interest", "Loyalty Prog."]

        fig_radar = go.Figure()
        for c in range(optimal_k):
            vals = dfc[dfc["Cluster"] == c][radar_features].mean().values
            # Normalize 0-1
            mins = dfc[radar_features].min().values
            maxs = dfc[radar_features].max().values
            vals_norm = (vals - mins) / (maxs - mins + 1e-8)
            fig_radar.add_trace(go.Scatterpolar(
                r=list(vals_norm) + [vals_norm[0]],
                theta=radar_labels + [radar_labels[0]],
                name=cluster_names[c], fill="toself", opacity=0.5
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Cluster Centroid Comparison (Normalized)",
            margin=dict(t=50, b=30, l=60, r=60), height=500
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.session_state["kmeans_model"] = km_final
        st.session_state["kmeans_scaler"] = scaler
        st.session_state["kmeans_features"] = feat_names
        st.session_state["cluster_names"] = cluster_names

    # ==================== LCA ====================
    with pred_tab3:
        st.subheader("Latent Class Analysis (LCA) — Hidden Consumer Typologies")
        st.markdown("""
        *Identifying latent segments from categorical patterns — nationality, language, 
        values, festival triggers, and ingredient priorities.*
        """)

        # Simplified LCA using K-Modes-like approach with label encoding + KMeans
        lca_cols = [
            "Q3_Nationality", "Q6_Language", "Q11_Brand_Loyalty",
            "Q15_Halal_Certified", "Q15_Vegan_Cruelty_Free",
            "Q15_Arabian_Heritage", "Q21_Influencer_Impact"
        ]
        lca_display = [
            "Nationality", "Language", "Brand Loyalty",
            "Halal Importance", "Vegan/CF Importance",
            "Arabian Heritage Imp.", "Influencer Impact"
        ]

        lca_df = df[lca_cols].copy()
        le_dict = {}
        lca_encoded = pd.DataFrame()
        for col in lca_cols:
            le = LabelEncoder()
            lca_encoded[col] = le.fit_transform(lca_df[col])
            le_dict[col] = le

        n_classes = st.slider("Number of Latent Classes:", 2, 6, 3, key="lca_n")

        km_lca = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
        lca_df["Latent_Class"] = km_lca.fit_predict(lca_encoded)

        class_names_map = {
            0: "Halal-Heritage Ritualists",
            1: "Globalised Clean-Beauty Seekers",
            2: "Price-Pragmatic Mainstream",
            3: "Eco-Conscious Trendsetters",
            4: "Status-Driven Luxury Buyers",
            5: "Digital-First Experimenters"
        }
        lca_df["Class_Name"] = lca_df["Latent_Class"].map(
            {k: class_names_map.get(k, f"Class {k}") for k in range(n_classes)}
        )

        # Class distribution
        class_dist = lca_df["Class_Name"].value_counts().reset_index()
        class_dist.columns = ["Latent Class", "Count"]
        fig_ld = px.bar(class_dist, x="Latent Class", y="Count",
                        title="Latent Class Distribution",
                        color="Latent Class",
                        color_discrete_sequence=px.colors.qualitative.Vivid)
        fig_ld.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=350, showlegend=False)
        st.plotly_chart(fig_ld, use_container_width=True)

        # Class profiles
        st.subheader("Latent Class Profiles")
        for cls in range(n_classes):
            cls_name = class_names_map.get(cls, f"Class {cls}")
            subset = lca_df[lca_df["Latent_Class"] == cls]
            with st.expander(f"📌 {cls_name} (n={len(subset)}, {len(subset)/len(df)*100:.1f}%)"):
                profile_cols = st.columns(len(lca_cols))
                for j, (col, display) in enumerate(zip(lca_cols, lca_display)):
                    with profile_cols[j % len(profile_cols)]:
                        top = subset[col].value_counts().head(3)
                        st.markdown(f"**{display}**")
                        for val, cnt in top.items():
                            pct = cnt / len(subset) * 100
                            short_val = val[:25] + "..." if len(str(val)) > 25 else val
                            st.caption(f"{short_val}: {pct:.0f}%")

        st.session_state["lca_model"] = km_lca
        st.session_state["lca_encoders"] = le_dict
        st.session_state["lca_cols"] = lca_cols
        st.session_state["lca_class_names"] = class_names_map
