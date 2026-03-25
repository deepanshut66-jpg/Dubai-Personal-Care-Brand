import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
from utils import add_numeric_columns, multiselect_to_binary


def render(df):
    st.header("🔍 Diagnostic Analytics — Why Is It Happening?")
    st.markdown("*Uncovering relationships, associations, and the drivers behind consumer spending.*")

    dfc = add_numeric_columns(df)

    diag_tab1, diag_tab2, diag_tab3 = st.tabs([
        "📦 Association Rule Mining", "📈 Regression Analysis", "📊 Cross-Tabulations"
    ])

    # ==================== ASSOCIATION RULES ====================
    with diag_tab1:
        st.subheader("Association Rule Mining (Apriori Algorithm)")
        st.markdown("""
        Discovering co-purchase patterns and hidden relationships between products, channels, 
        and preferences to inform bundling and cross-sell strategies.
        """)

        arm_source = st.selectbox("Select itemset source:", [
            "Product Categories (Q14)",
            "Channels (Q9)",
            "Discovery Methods (Q20)",
            "Festival Triggers (Q17)"
        ])

        col_map = {
            "Product Categories (Q14)": "Q14_Product_Categories",
            "Channels (Q9)": "Q9_Channels",
            "Discovery Methods (Q20)": "Q20_Discovery_Methods",
            "Festival Triggers (Q17)": "Q17_Festival_Triggers"
        }
        selected_col = col_map[arm_source]

        min_sup = st.slider("Minimum Support", 0.05, 0.40, 0.10, 0.01, key="arm_support")
        min_conf = st.slider("Minimum Confidence", 0.10, 0.80, 0.30, 0.05, key="arm_conf")

        binary_df = multiselect_to_binary(df, selected_col)

        if binary_df.shape[1] < 2:
            st.warning("Not enough items for association mining.")
        else:
            freq_items = apriori(binary_df, min_support=min_sup, use_colnames=True)

            if len(freq_items) == 0:
                st.warning("No frequent itemsets found. Try lowering the minimum support.")
            else:
                rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)

                if len(rules) == 0:
                    st.warning("No rules found at this confidence level. Try lowering the threshold.")
                else:
                    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
                    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

                    display_rules = rules[["antecedents_str", "consequents_str",
                                           "support", "confidence", "lift"]].copy()
                    display_rules.columns = ["If Consumer Buys...", "They Also Buy...",
                                             "Support", "Confidence", "Lift"]
                    display_rules = display_rules.sort_values("Lift", ascending=False).head(20)

                    st.dataframe(
                        display_rules.style.format({
                            "Support": "{:.3f}", "Confidence": "{:.3f}", "Lift": "{:.2f}"
                        }).background_gradient(subset=["Lift"], cmap="YlGn"),
                        use_container_width=True, height=400
                    )

                    st.markdown("**Key Metrics:**")
                    st.markdown("""
                    - **Support**: How frequently the items appear together (higher = more common)
                    - **Confidence**: Probability of consequent given antecedent (higher = stronger rule)
                    - **Lift**: How much more likely they co-occur vs. random (>1 = positive association)
                    """)

                    # Scatter plot
                    fig = px.scatter(rules, x="support", y="confidence", size="lift",
                                     color="lift", color_continuous_scale="Tealgrn",
                                     hover_data=["antecedents_str", "consequents_str"],
                                     title="Association Rules: Support vs Confidence (Size = Lift)")
                    fig.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=450)
                    st.plotly_chart(fig, use_container_width=True)

    # ==================== REGRESSION ====================
    with diag_tab2:
        st.subheader("Multiple Linear Regression — Spending Drivers")
        st.markdown("*What factors drive how much a consumer spends on personal care per month?*")

        feature_names = [
            "Age_Numeric", "Freq_Numeric", "Loyalty_Numeric", "Premium_Numeric",
            "Influencer_Numeric", "AI_Numeric", "Gift_Numeric", "LoyaltyProg_Numeric",
            "Q15_Halal_Certified_Num", "Q15_Natural_Organic_Num",
            "Q15_Dermatologist_Tested_Num", "Q15_Vegan_Cruelty_Free_Num"
        ]
        display_names = [
            "Age", "Purchase Frequency", "Brand Loyalty", "Price Premium Willingness",
            "Influencer Impact", "AI Tool Interest", "Gift Purchases", "Loyalty Programme",
            "Halal Importance", "Natural/Organic Imp.", "Derm-Tested Imp.", "Vegan/CF Imp."
        ]

        reg_df = dfc[feature_names + ["Spend_Numeric", "Income_Numeric"]].dropna()
        feature_names_with_income = feature_names + ["Income_Numeric"]
        display_names_with_income = display_names + ["Income"]

        X_reg = reg_df[feature_names_with_income]
        y_reg = reg_df["Spend_Numeric"]

        X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        r2_train = model.score(X_train, y_train)
        r2_test = model.score(X_test, y_test)

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("R² (Train)", f"{r2_train:.4f}")
        rc2.metric("R² (Test)", f"{r2_test:.4f}")
        rc3.metric("Intercept", f"{model.intercept_:.2f}")

        coef_df = pd.DataFrame({
            "Feature": display_names_with_income,
            "Coefficient": model.coef_
        }).sort_values("Coefficient", key=abs, ascending=True)

        fig = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                     title="Regression Coefficients (Impact on Monthly Spend)",
                     color="Coefficient", color_continuous_scale="RdBu_r")
        fig.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Residual plot
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        fig2 = px.scatter(x=y_pred, y=residuals, title="Residual Plot",
                          labels={"x": "Predicted Spend (AED)", "y": "Residual"},
                          opacity=0.4, color_discrete_sequence=["#0D7377"])
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        fig2.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=400)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Interpretation**: Positive coefficients increase monthly spend; negative decrease it. "
                    "Income and Purchase Frequency are typically the strongest predictors.")

    # ==================== CROSS-TABS ====================
    with diag_tab3:
        st.subheader("Cross-Tabulation Analysis")
        st.markdown("*Exploring how demographics interact with purchasing behaviour.*")

        row_var = st.selectbox("Row Variable:", [
            "Q3_Nationality", "Q1_Age_Group", "Q2_Gender", "Q5_Income"
        ], key="xtab_row")
        col_var = st.selectbox("Column Variable:", [
            "Q8_Monthly_Spend", "Q11_Brand_Loyalty", "Q13_Price_Premium",
            "Q25_Brand_Interest", "Q21_Influencer_Impact"
        ], key="xtab_col")

        ct = pd.crosstab(df[row_var], df[col_var], normalize="index") * 100
        fig = px.imshow(ct, text_auto=".1f", color_continuous_scale="Tealgrn",
                        title=f"Cross-Tab: {row_var} × {col_var} (Row %)",
                        aspect="auto")
        fig.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Grouped bar
        ct2 = pd.crosstab(df[row_var], df[col_var]).reset_index().melt(
            id_vars=row_var, var_name=col_var, value_name="Count"
        )
        fig2 = px.bar(ct2, x=row_var, y="Count", color=col_var,
                      title=f"Count: {row_var} × {col_var}",
                      barmode="group", color_discrete_sequence=px.colors.qualitative.Teal_r)
        fig2.update_layout(margin=dict(t=50, b=30, l=30, r=30), height=450,
                           xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)
