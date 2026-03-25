import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

SPEND_MAP = {
    "Below 50": 25, "50-100": 75, "101-200": 150,
    "201-400": 300, "401-700": 550, "Above 700": 900
}

INCOME_MAP = {
    "Below 5,000": 3000, "5,000-10,000": 7500, "10,001-20,000": 15000,
    "20,001-35,000": 27500, "35,001-50,000": 42500, "Above 50,000": 65000,
    "Prefer not to say": np.nan
}

AGE_MAP = {"18-24": 1, "25-34": 2, "35-44": 3, "45-54": 4, "55+": 5}

FREQ_MAP = {
    "Weekly": 5, "Every 2 weeks": 4, "Monthly": 3,
    "Every 2-3 months": 2, "Rarely / Only when needed": 1
}

LOYALTY_MAP = {"Very loyal": 5, "Somewhat loyal": 4, "Neutral": 3, "Low loyalty": 2, "No loyalty": 1}

PREMIUM_MAP = {
    "0% - would not pay more": 0, "Up to 10% more": 1,
    "11-25% more": 2, "26-50% more": 3, "More than 50% more": 4
}

INFLUENCER_MAP = {"Very strongly": 4, "Moderately": 3, "Slightly": 2, "Not at all": 1}

AI_MAP = {"Yes, definitely": 4, "Probably": 3, "Unsure": 2, "No": 1}

GIFT_MAP = {
    "Yes, frequently (5+ times/year)": 4, "Yes, occasionally (2-4 times/year)": 3,
    "Rarely (once a year or less)": 2, "Never": 1
}

LOYALTY_PROG_MAP = {"Very likely": 5, "Likely": 4, "Neutral": 3, "Unlikely": 2, "Very unlikely": 1}

IMPORTANCE_MAP = {"Not Important": 1, "Somewhat Important": 2, "Very Important": 3, "Essential": 4}

INTEREST_BINARY = {
    "Very interested": 1, "Interested": 1,
    "Neutral": 0, "Not very interested": 0, "Not interested at all": 0
}


def load_data(path="dubai_personal_care_survey.csv"):
    return pd.read_csv(path)


def add_numeric_columns(df):
    dfc = df.copy()
    dfc["Spend_Numeric"] = dfc["Q8_Monthly_Spend"].map(SPEND_MAP)
    dfc["Income_Numeric"] = dfc["Q5_Income"].map(INCOME_MAP)
    dfc["Age_Numeric"] = dfc["Q1_Age_Group"].map(AGE_MAP)
    dfc["Freq_Numeric"] = dfc["Q7_Purchase_Frequency"].map(FREQ_MAP)
    dfc["Loyalty_Numeric"] = dfc["Q11_Brand_Loyalty"].map(LOYALTY_MAP)
    dfc["Premium_Numeric"] = dfc["Q13_Price_Premium"].map(PREMIUM_MAP)
    dfc["Influencer_Numeric"] = dfc["Q21_Influencer_Impact"].map(INFLUENCER_MAP)
    dfc["AI_Numeric"] = dfc["Q22_AI_Tool_Interest"].map(AI_MAP)
    dfc["Gift_Numeric"] = dfc["Q18_Gift_Purchases"].map(GIFT_MAP)
    dfc["LoyaltyProg_Numeric"] = dfc["Q24_Loyalty_Programme"].map(LOYALTY_PROG_MAP)
    for col in ["Q15_Halal_Certified", "Q15_Natural_Organic", "Q15_Free_From_Chemicals",
                 "Q15_Dermatologist_Tested", "Q15_Vegan_Cruelty_Free", "Q15_Arabian_Heritage"]:
        dfc[col + "_Num"] = dfc[col].map(IMPORTANCE_MAP)
    dfc["Interest_Binary"] = dfc["Q25_Brand_Interest"].map(INTEREST_BINARY)
    return dfc


def get_clustering_features(df_numeric):
    features = [
        "Age_Numeric", "Spend_Numeric", "Freq_Numeric", "Loyalty_Numeric",
        "Premium_Numeric", "Influencer_Numeric", "AI_Numeric", "Gift_Numeric",
        "LoyaltyProg_Numeric",
        "Q15_Halal_Certified_Num", "Q15_Natural_Organic_Num",
        "Q15_Free_From_Chemicals_Num", "Q15_Dermatologist_Tested_Num",
        "Q15_Vegan_Cruelty_Free_Num", "Q15_Arabian_Heritage_Num"
    ]
    X = df_numeric[features].copy()
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, features, scaler


def get_classification_features(df_numeric):
    feature_cols = [
        "Age_Numeric", "Spend_Numeric", "Freq_Numeric", "Loyalty_Numeric",
        "Premium_Numeric", "Influencer_Numeric", "AI_Numeric", "Gift_Numeric",
        "LoyaltyProg_Numeric",
        "Q15_Halal_Certified_Num", "Q15_Natural_Organic_Num",
        "Q15_Free_From_Chemicals_Num", "Q15_Dermatologist_Tested_Num",
        "Q15_Vegan_Cruelty_Free_Num", "Q15_Arabian_Heritage_Num"
    ]
    le_gender = LabelEncoder()
    df_numeric["Gender_Enc"] = le_gender.fit_transform(df_numeric["Q2_Gender"])
    feature_cols.append("Gender_Enc")

    le_nat = LabelEncoder()
    df_numeric["Nationality_Enc"] = le_nat.fit_transform(df_numeric["Q3_Nationality"])
    feature_cols.append("Nationality_Enc")

    le_reason = LabelEncoder()
    df_numeric["Reason_Enc"] = le_reason.fit_transform(df_numeric["Q10_Channel_Reason"])
    feature_cols.append("Reason_Enc")

    le_disc = LabelEncoder()
    df_numeric["Discount_Enc"] = le_disc.fit_transform(df_numeric["Q23_Discount_Type"])
    feature_cols.append("Discount_Enc")

    X = df_numeric[feature_cols].fillna(0)
    y = df_numeric["Interest_Binary"]

    encoders = {
        "gender": le_gender, "nationality": le_nat,
        "reason": le_reason, "discount": le_disc
    }
    return X, y, feature_cols, encoders


def explode_multiselect(df, column):
    items = df[column].str.split("; ").explode().str.strip()
    return items[items != ""].value_counts()


def multiselect_to_binary(df, column):
    items = set()
    for val in df[column].dropna():
        for item in str(val).split("; "):
            item = item.strip()
            if item:
                items.add(item)
    items = sorted(items)
    binary_df = pd.DataFrame(False, index=df.index, columns=items)
    for idx, val in df[column].dropna().items():
        for item in str(val).split("; "):
            item = item.strip()
            if item in binary_df.columns:
                binary_df.at[idx, item] = True
    return binary_df
