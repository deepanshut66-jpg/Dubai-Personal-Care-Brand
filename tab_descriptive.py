import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils import add_numeric_columns, explode_multiselect, SPEND_MAP, IMPORTANCE_MAP


def render(df):
    st.header("📊 Descriptive Analytics — Market Snapshot")
    st.markdown("*Understanding who our consumers are, what they buy, and how they behave.*")

    dfc = add_numeric_columns(df)

    st.subheader("1. Demographic Profile")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(df, names="Q1_Age_Group", title="Age Distribution",
                     color_discrete_sequence=px.colors.qualitative.Teal_r,
                     hole=0.4)
        fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(df, names="Q2_Gender", title="Gender Distribution",
                     color_discrete_sequence=px.colors.qualitative.Safe,
                     hole=0.4)
        fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        nat_counts = df["Q3_Nationality"].value_counts().reset_index()
        nat_counts.columns = ["Nationality", "Count"]
        fig = px.bar(nat_counts, x="Count", y="Nationality", orientation="h",
                     title="Nationality Breakdown", color="Count",
                     color_continuous_scale="Tealgrn")
        fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        inc_order = ["Below 5,000", "5,000-10,000", "10,001-20,000",
                     "20,001-35,000", "35,001-50,000", "Above 50,000", "Prefer not to say"]
        inc_counts = df["Q5_Income"].value_counts().reindex(inc_order).reset_index()
        inc_counts.columns = ["Income (AED)", "Count"]
        fig = px.bar(inc_counts, x="Income (AED)", y="Count",
                     title="Income Distribution", color="Count",
                     color_continuous_scale="Purp")
        fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("2. Purchase Behaviour")
    c5, c6 = st.columns(2)
    with c5:
        spend_order = list(SPEND_MAP.keys())
        spend_counts = df["Q8_Monthly_Spend"].value_counts().reindex(spend_order).reset_index()
        spend_counts.columns = ["Monthly Spend (AED)", "Count"]
        fig = px.bar(spend_counts, x="Monthly Spend (AED)", y="Count",
                     title="Monthly Spend Distribution", color="Count",
                     color_continuous_scale="Tealgrn")
        fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        freq_order = ["Weekly", "Every 2 weeks", "Monthly", "Every 2-3 months", "Rarely / Only when needed"]
        freq_counts = df["Q7_Purchase_Frequency"].value_counts().reindex(freq_order).reset_index()
        freq_counts.columns = ["Frequency", "Count"]
        fig = px.bar(freq_counts, x="Frequency", y="Count",
                     title="Purchase Frequency", color="Count",
                     color_continuous_scale="Purp")
        fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Channel preferences
    st.subheader("3. Channel Preferences")
    ch_counts = explode_multiselect(df, "Q9_Channels").reset_index()
    ch_counts.columns = ["Channel", "Count"]
    fig = px.bar(ch_counts.sort_values("Count", ascending=True), x="Count", y="Channel",
                 orientation="h", title="Most Used Channels (Multi-Select)",
                 color="Count", color_continuous_scale="Tealgrn")
    fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Product categories
    st.subheader("4. Product Category Demand")
    cat_counts = explode_multiselect(df, "Q14_Product_Categories").reset_index()
    cat_counts.columns = ["Category", "Count"]
    fig = px.bar(cat_counts.sort_values("Count", ascending=True), x="Count", y="Category",
                 orientation="h", title="Product Categories Purchased (Multi-Select)",
                 color="Count", color_continuous_scale="Purp")
    fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Ingredient Heatmap
    st.subheader("5. Ingredient Importance Heatmap by Nationality")
    ing_cols = ["Q15_Halal_Certified", "Q15_Natural_Organic", "Q15_Free_From_Chemicals",
                "Q15_Dermatologist_Tested", "Q15_Vegan_Cruelty_Free", "Q15_Arabian_Heritage"]
    ing_labels = ["Halal", "Natural/Organic", "Chemical-Free", "Derm-Tested", "Vegan/CF", "Arabian Heritage"]
    heat_data = []
    for nat in df["Q3_Nationality"].unique():
        subset = dfc[dfc["Q3_Nationality"] == nat]
        row = []
        for col in ing_cols:
            row.append(subset[col].map(IMPORTANCE_MAP).mean())
        heat_data.append(row)
    heat_df = pd.DataFrame(heat_data, index=df["Q3_Nationality"].unique(), columns=ing_labels)
    heat_df = heat_df.sort_index()
    fig = px.imshow(heat_df, text_auto=".2f", color_continuous_scale="Tealgrn",
                    title="Average Ingredient Importance (1=Not Important, 4=Essential)",
                    aspect="auto")
    fig.update_layout(margin=dict(t=50, b=20, l=20, r=20), height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Brand Interest
    st.subheader("6. Brand Interest (Target Variable)")
    interest_order = ["Very interested", "Interested", "Neutral", "Not very interested", "Not interested at all"]
    int_counts = df["Q25_Brand_Interest"].value_counts().reindex(interest_order).reset_index()
    int_counts.columns = ["Interest Level", "Count"]
    fig = px.bar(int_counts, x="Interest Level", y="Count",
                 title="Overall Brand Interest Distribution",
                 color="Interest Level",
                 color_discrete_sequence=["#0D7377", "#14919B", "#9DC5BB", "#E8927C", "#D94F3B"])
    fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Key Metrics Summary
    st.subheader("7. Key Metrics Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Respondents", f"{len(df):,}")
    interested = dfc["Interest_Binary"].sum()
    m2.metric("% Interested", f"{interested / len(df) * 100:.1f}%")
    m3.metric("Avg Monthly Spend", f"AED {dfc['Spend_Numeric'].mean():.0f}")
    top_channel = explode_multiselect(df, "Q9_Channels").index[0]
    m4.metric("Top Channel", top_channel.split("(")[0].strip())
