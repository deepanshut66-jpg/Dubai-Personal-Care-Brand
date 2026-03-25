import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils import (add_numeric_columns, get_clustering_features, SPEND_MAP, INCOME_MAP,
                   AGE_MAP, FREQ_MAP, LOYALTY_MAP, PREMIUM_MAP, INFLUENCER_MAP,
                   AI_MAP, GIFT_MAP, LOYALTY_PROG_MAP, IMPORTANCE_MAP, INTEREST_BINARY)
import plotly.express as px


def render(df):
    st.header("🆕 New Customer Predictor")
    st.markdown("""
    *Upload new survey data to instantly predict brand interest, assign clusters, 
    and generate targeted marketing recommendations.*
    """)

    mode = st.radio("Choose input mode:", ["Upload CSV File", "Single Customer Entry"], horizontal=True)

    if mode == "Upload CSV File":
        render_upload(df)
    else:
        render_single(df)


def render_upload(df):
    st.markdown("""
    **Upload a CSV** with the same column structure as the original survey data.  
    The system will predict brand interest, assign K-Means cluster, and recommend strategies.
    """)

    uploaded = st.file_uploader("Upload new customer survey data (.csv)", type=["csv"])

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(new_df)} records with {len(new_df.columns)} columns.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        if "best_model" not in st.session_state:
            st.warning("⚠️ Please run the **Predictive Analytics → Classification** tab first.")
            return
        if "kmeans_model" not in st.session_state:
            st.warning("⚠️ Please run the **Predictive Analytics → K-Means Clustering** tab first.")
            return

        try:
            predictions = predict_batch(new_df, df)
            st.subheader("Prediction Results")

            # Summary metrics
            m1, m2, m3 = st.columns(3)
            interested_pct = predictions["Predicted_Interest"].mean() * 100
            m1.metric("Predicted Interested %", f"{interested_pct:.1f}%")
            m2.metric("Total New Customers", f"{len(predictions):,}")
            avg_prob = predictions["Interest_Probability"].mean() * 100
            m3.metric("Avg Interest Probability", f"{avg_prob:.1f}%")

            # Distribution
            fig = px.histogram(predictions, x="Interest_Probability", nbins=20,
                               title="Interest Probability Distribution",
                               color_discrete_sequence=["#0D7377"])
            fig.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=350)
            st.plotly_chart(fig, use_container_width=True)

            # Cluster distribution
            if "Assigned_Cluster" in predictions.columns:
                cluster_dist = predictions["Assigned_Cluster"].value_counts().reset_index()
                cluster_dist.columns = ["Segment", "Count"]
                fig2 = px.pie(cluster_dist, values="Count", names="Segment",
                              title="New Customers by Segment", hole=0.4,
                              color_discrete_sequence=px.colors.qualitative.Bold)
                fig2.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=350)
                st.plotly_chart(fig2, use_container_width=True)

            # Full table
            st.subheader("Detailed Predictions")
            st.dataframe(predictions.style.format({
                "Interest_Probability": "{:.2%}"
            }), use_container_width=True, height=400)

            # Download
            csv = predictions.to_csv(index=False)
            st.download_button(
                "📥 Download Predictions CSV",
                data=csv, file_name="new_customer_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Make sure the uploaded CSV has the same column names as the original survey data.")


def render_single(df):
    st.markdown("**Enter a single customer's survey responses to get instant predictions.**")

    with st.form("single_customer"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.selectbox("Q1: Age Group", ["18-24", "25-34", "35-44", "45-54", "55+"])
            gender = st.selectbox("Q2: Gender", ["Male", "Female", "Non-binary / Prefer not to say"])
            nationality = st.selectbox("Q3: Nationality", [
                "South Asian (Indian, Pakistani, Bangladeshi, Sri Lankan)",
                "Emirati", "Other Arab (Egyptian, Lebanese, Jordanian, etc.)",
                "Filipino", "Western Expat (European, American, Australian)",
                "East Asian (Chinese, Korean, Japanese)", "African", "Other"
            ])
        with c2:
            income = st.selectbox("Q5: Income (AED)", [
                "Below 5,000", "5,000-10,000", "10,001-20,000",
                "20,001-35,000", "35,001-50,000", "Above 50,000"
            ])
            spend = st.selectbox("Q8: Monthly Spend (AED)", list(SPEND_MAP.keys()))
            frequency = st.selectbox("Q7: Purchase Frequency", [
                "Weekly", "Every 2 weeks", "Monthly", "Every 2-3 months", "Rarely / Only when needed"
            ])
        with c3:
            loyalty = st.selectbox("Q11: Brand Loyalty", [
                "Very loyal", "Somewhat loyal", "Neutral", "Low loyalty", "No loyalty"
            ])
            premium = st.selectbox("Q13: Price Premium", [
                "0% - would not pay more", "Up to 10% more", "11-25% more",
                "26-50% more", "More than 50% more"
            ])
            influencer = st.selectbox("Q21: Influencer Impact", [
                "Very strongly", "Moderately", "Slightly", "Not at all"
            ])

        c4, c5 = st.columns(2)
        with c4:
            ai_tool = st.selectbox("Q22: AI Tool Interest", [
                "Yes, definitely", "Probably", "Unsure", "No"
            ])
            gift = st.selectbox("Q18: Gift Purchases", [
                "Yes, frequently (5+ times/year)", "Yes, occasionally (2-4 times/year)",
                "Rarely (once a year or less)", "Never"
            ])
            loyalty_prog = st.selectbox("Q24: Loyalty Programme", [
                "Very likely", "Likely", "Neutral", "Unlikely", "Very unlikely"
            ])
        with c5:
            halal = st.selectbox("Q15: Halal Importance", list(IMPORTANCE_MAP.keys()))
            natural = st.selectbox("Q15: Natural/Organic Importance", list(IMPORTANCE_MAP.keys()))
            vegan = st.selectbox("Q15: Vegan/CF Importance", list(IMPORTANCE_MAP.keys()))
            channel_reason = st.selectbox("Q10: Channel Reason", [
                "Best prices / discounts", "Widest product range",
                "Trusted recommendations / expert staff", "Convenience / delivery speed",
                "Ability to test / try before buying", "Loyalty rewards / cashback",
                "Exclusive products not available elsewhere"
            ])
            discount = st.selectbox("Q23: Discount Type", [
                "Percentage discount (e.g., 20% off)", "Buy-one-get-one-free (BOGO)",
                "Free sample / trial size", "Loyalty points", "Bundle deal",
                "Free shipping", "Subscription discount"
            ])

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        if "best_model" not in st.session_state:
            st.warning("⚠️ Please run the **Predictive Analytics** tab first.")
            return

        row = {
            "Age_Numeric": AGE_MAP[age],
            "Spend_Numeric": SPEND_MAP[spend],
            "Freq_Numeric": FREQ_MAP[frequency],
            "Loyalty_Numeric": LOYALTY_MAP[loyalty],
            "Premium_Numeric": PREMIUM_MAP[premium],
            "Influencer_Numeric": INFLUENCER_MAP[influencer],
            "AI_Numeric": AI_MAP[ai_tool],
            "Gift_Numeric": GIFT_MAP[gift],
            "LoyaltyProg_Numeric": LOYALTY_PROG_MAP[loyalty_prog],
            "Q15_Halal_Certified_Num": IMPORTANCE_MAP[halal],
            "Q15_Natural_Organic_Num": IMPORTANCE_MAP[natural],
            "Q15_Free_From_Chemicals_Num": IMPORTANCE_MAP.get("Somewhat Important", 2),
            "Q15_Dermatologist_Tested_Num": IMPORTANCE_MAP.get("Somewhat Important", 2),
            "Q15_Vegan_Cruelty_Free_Num": IMPORTANCE_MAP[vegan],
            "Q15_Arabian_Heritage_Num": IMPORTANCE_MAP.get("Somewhat Important", 2),
        }

        # Encode categoricals
        encoders = st.session_state["encoders"]
        for val, key, enc_name in [
            (gender, "Gender_Enc", "gender"),
            (nationality, "Nationality_Enc", "nationality"),
            (channel_reason, "Reason_Enc", "reason"),
            (discount, "Discount_Enc", "discount")
        ]:
            enc = encoders[enc_name]
            if val in enc.classes_:
                row[key] = enc.transform([val])[0]
            else:
                row[key] = 0

        feature_cols = st.session_state["feature_cols"]
        X_new = pd.DataFrame([row])[feature_cols].fillna(0)
        model = st.session_state["best_model"]
        pred = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0][1]

        # Cluster assignment
        cluster_label = "N/A"
        if "kmeans_model" in st.session_state:
            km = st.session_state["kmeans_model"]
            km_scaler = st.session_state["kmeans_scaler"]
            km_features = st.session_state["kmeans_features"]
            cluster_names = st.session_state["cluster_names"]
            km_row = {f: row.get(f, 0) for f in km_features}
            X_km = pd.DataFrame([km_row])[km_features].fillna(0)
            X_km_scaled = km_scaler.transform(X_km)
            cl = km.predict(X_km_scaled)[0]
            cluster_label = cluster_names.get(cl, f"Cluster {cl}")

        st.divider()
        st.subheader("Prediction Results")
        r1, r2, r3 = st.columns(3)
        r1.metric("Brand Interest", "✅ Interested" if pred == 1 else "❌ Not Interested")
        r2.metric("Interest Probability", f"{prob:.1%}")
        r3.metric("Assigned Segment", cluster_label)

        if pred == 1:
            st.success(f"This customer has a **{prob:.0%}** probability of being interested. "
                       f"Assign to the **{cluster_label}** segment and use the corresponding "
                       f"marketing playbook from the Prescriptive tab.")
        else:
            st.info(f"This customer has a lower probability ({prob:.0%}). Consider a "
                    f"low-cost awareness campaign rather than direct conversion efforts.")


def predict_batch(new_df, original_df):
    dfc_new = add_numeric_columns(new_df)
    model = st.session_state["best_model"]
    feature_cols = st.session_state["feature_cols"]
    encoders = st.session_state["encoders"]

    # Encode categoricals
    for col, enc_name, src_col in [
        ("Gender_Enc", "gender", "Q2_Gender"),
        ("Nationality_Enc", "nationality", "Q3_Nationality"),
        ("Reason_Enc", "reason", "Q10_Channel_Reason"),
        ("Discount_Enc", "discount", "Q23_Discount_Type"),
    ]:
        enc = encoders[enc_name]
        dfc_new[col] = dfc_new[src_col].apply(
            lambda x: enc.transform([x])[0] if x in enc.classes_ else 0
        )

    X_new = dfc_new[feature_cols].fillna(0)
    predictions = pd.DataFrame()
    predictions["Predicted_Interest"] = model.predict(X_new)
    predictions["Interest_Probability"] = model.predict_proba(X_new)[:, 1]

    # K-Means cluster
    if "kmeans_model" in st.session_state:
        km = st.session_state["kmeans_model"]
        km_scaler = st.session_state["kmeans_scaler"]
        km_features = st.session_state["kmeans_features"]
        cluster_names = st.session_state["cluster_names"]
        X_km = dfc_new[km_features].fillna(0)
        X_km_scaled = km_scaler.transform(X_km)
        predictions["Assigned_Cluster"] = [cluster_names.get(c, f"Cluster {c}")
                                            for c in km.predict(X_km_scaled)]

    # Add key original columns for context
    for col in ["Q1_Age_Group", "Q2_Gender", "Q3_Nationality", "Q5_Income", "Q8_Monthly_Spend"]:
        if col in new_df.columns:
            predictions[col] = new_df[col].values

    return predictions
