import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 5000

# --- Q1: Age Group ---
age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
age_probs = [0.22, 0.35, 0.25, 0.12, 0.06]
q1 = np.random.choice(age_groups, N, p=age_probs)

# --- Q2: Gender ---
genders = ["Male", "Female", "Non-binary / Prefer not to say"]
gender_probs = [0.62, 0.35, 0.03]
q2 = np.random.choice(genders, N, p=gender_probs)

# --- Q3: Nationality ---
nationalities = [
    "South Asian (Indian, Pakistani, Bangladeshi, Sri Lankan)",
    "Emirati",
    "Other Arab (Egyptian, Lebanese, Jordanian, etc.)",
    "Filipino",
    "Western Expat (European, American, Australian)",
    "East Asian (Chinese, Korean, Japanese)",
    "African",
    "Other"
]
nat_probs = [0.38, 0.10, 0.15, 0.08, 0.12, 0.07, 0.06, 0.04]
q3 = np.random.choice(nationalities, N, p=nat_probs)

# --- Q4: Area ---
areas = [
    "Downtown / Business Bay / DIFC",
    "Dubai Marina / JBR / JLT",
    "Jumeirah / Umm Suqeim",
    "Deira / Bur Dubai / Al Karama",
    "Al Barsha / Discovery Gardens / JVC",
    "Silicon Oasis / Academic City / Sports City",
    "Mirdif / Al Warqa / Rashidiya",
    "Other"
]
area_probs = [0.15, 0.18, 0.08, 0.20, 0.15, 0.10, 0.09, 0.05]
q4 = np.random.choice(areas, N, p=area_probs)

# --- Q5: Income (correlated with nationality) ---
income_levels = [
    "Below 5,000", "5,000-10,000", "10,001-20,000",
    "20,001-35,000", "35,001-50,000", "Above 50,000", "Prefer not to say"
]
income_map = {
    "South Asian (Indian, Pakistani, Bangladeshi, Sri Lankan)": [0.08, 0.22, 0.32, 0.22, 0.08, 0.04, 0.04],
    "Emirati": [0.01, 0.03, 0.10, 0.25, 0.30, 0.28, 0.03],
    "Other Arab (Egyptian, Lebanese, Jordanian, etc.)": [0.05, 0.15, 0.30, 0.25, 0.14, 0.07, 0.04],
    "Filipino": [0.10, 0.30, 0.32, 0.18, 0.05, 0.02, 0.03],
    "Western Expat (European, American, Australian)": [0.02, 0.04, 0.12, 0.28, 0.28, 0.22, 0.04],
    "East Asian (Chinese, Korean, Japanese)": [0.03, 0.08, 0.20, 0.30, 0.22, 0.13, 0.04],
    "African": [0.12, 0.25, 0.30, 0.20, 0.07, 0.03, 0.03],
    "Other": [0.06, 0.15, 0.25, 0.25, 0.15, 0.10, 0.04],
}
q5 = np.array([np.random.choice(income_levels, p=income_map[n]) for n in q3])

# --- Q6: Language ---
lang_map = {
    "South Asian (Indian, Pakistani, Bangladeshi, Sri Lankan)": ["English", "Hindi / Urdu", "Hindi / Urdu", "English", "English"],
    "Emirati": ["Arabic", "Arabic", "Arabic", "English", "Arabic"],
    "Other Arab (Egyptian, Lebanese, Jordanian, etc.)": ["Arabic", "Arabic", "English", "Arabic", "English"],
    "Filipino": ["English", "Tagalog / Filipino", "English", "Tagalog / Filipino", "English"],
    "Western Expat (European, American, Australian)": ["English", "English", "English", "English", "Other"],
    "East Asian (Chinese, Korean, Japanese)": ["English", "Mandarin / Other Asian language", "English", "Mandarin / Other Asian language", "English"],
    "African": ["English", "English", "Arabic", "English", "Other"],
    "Other": ["English", "English", "Arabic", "English", "Other"],
}
q6 = np.array([np.random.choice(lang_map[n]) for n in q3])

# --- Q7: Purchase Frequency (correlated with income and age) ---
freq_options = ["Weekly", "Every 2 weeks", "Monthly", "Every 2-3 months", "Rarely / Only when needed"]
freq_map_by_income = {
    "Below 5,000": [0.03, 0.08, 0.30, 0.38, 0.21],
    "5,000-10,000": [0.05, 0.10, 0.35, 0.35, 0.15],
    "10,001-20,000": [0.07, 0.15, 0.40, 0.28, 0.10],
    "20,001-35,000": [0.10, 0.20, 0.38, 0.24, 0.08],
    "35,001-50,000": [0.14, 0.25, 0.35, 0.18, 0.08],
    "Above 50,000": [0.20, 0.28, 0.30, 0.15, 0.07],
    "Prefer not to say": [0.08, 0.15, 0.38, 0.28, 0.11],
}
q7 = np.array([np.random.choice(freq_options, p=freq_map_by_income[inc]) for inc in q5])

# --- Q8: Monthly Spend (correlated with income) ---
spend_options = ["Below 50", "50-100", "101-200", "201-400", "401-700", "Above 700"]
spend_map = {
    "Below 5,000": [0.40, 0.35, 0.15, 0.06, 0.03, 0.01],
    "5,000-10,000": [0.20, 0.35, 0.28, 0.12, 0.04, 0.01],
    "10,001-20,000": [0.08, 0.22, 0.35, 0.25, 0.07, 0.03],
    "20,001-35,000": [0.04, 0.12, 0.28, 0.32, 0.16, 0.08],
    "35,001-50,000": [0.02, 0.06, 0.18, 0.30, 0.28, 0.16],
    "Above 50,000": [0.01, 0.03, 0.10, 0.22, 0.32, 0.32],
    "Prefer not to say": [0.10, 0.20, 0.30, 0.22, 0.12, 0.06],
}
q8 = np.array([np.random.choice(spend_options, p=spend_map[inc]) for inc in q5])

# --- Q9: Channels (multi-select, up to 3) ---
channels = [
    "Hypermarkets (Carrefour, Lulu, Spinneys)",
    "Pharmacies (BinSina, Aster, Life Pharmacy)",
    "Beauty specialist stores (Sephora, Faces, MAC)",
    "Online marketplaces (Amazon.ae, Noon)",
    "Brand websites / D2C apps",
    "Duty-free shops",
    "Baqala / convenience stores",
    "Subscription boxes"
]
ch_base = [0.55, 0.40, 0.30, 0.45, 0.18, 0.15, 0.20, 0.08]
q9 = []
for i in range(N):
    probs = ch_base.copy()
    if q5[i] in ["Below 5,000", "5,000-10,000"]:
        probs[0] += 0.15; probs[6] += 0.10; probs[2] -= 0.10
    if q5[i] in ["Above 50,000", "35,001-50,000"]:
        probs[2] += 0.15; probs[5] += 0.10; probs[4] += 0.10
    if q1[i] in ["18-24", "25-34"]:
        probs[3] += 0.12; probs[4] += 0.08; probs[7] += 0.05
    probs = np.clip(probs, 0.02, 0.95)
    selected = [ch for ch, p in zip(channels, probs) if np.random.random() < p]
    if len(selected) == 0:
        selected = [np.random.choice(channels)]
    if len(selected) > 3:
        selected = list(np.random.choice(selected, 3, replace=False))
    q9.append("; ".join(selected))
q9 = np.array(q9)

# --- Q10: Primary Channel Reason ---
reasons = [
    "Best prices / discounts",
    "Widest product range",
    "Trusted recommendations / expert staff",
    "Convenience / delivery speed",
    "Ability to test / try before buying",
    "Loyalty rewards / cashback",
    "Exclusive products not available elsewhere"
]
reason_probs = [0.22, 0.15, 0.12, 0.20, 0.13, 0.10, 0.08]
q10 = np.random.choice(reasons, N, p=reason_probs)

# --- Q11: Brand Loyalty ---
loyalty_options = [
    "Very loyal",
    "Somewhat loyal",
    "Neutral",
    "Low loyalty",
    "No loyalty"
]
loyalty_probs = [0.12, 0.32, 0.28, 0.20, 0.08]
q11 = np.random.choice(loyalty_options, N, p=loyalty_probs)

# --- Q12: Switching Triggers (multi-select up to 2) ---
switch_triggers = [
    "Lower price / better value",
    "Superior ingredients / formulation",
    "Recommendation from a friend or influencer",
    "Attractive packaging / brand story",
    "Halal / clean / organic certification",
    "Availability in my preferred store",
    "Personalised products"
]
sw_probs = [0.40, 0.30, 0.25, 0.15, 0.22, 0.18, 0.20]
q12 = []
for i in range(N):
    p = sw_probs.copy()
    if q11[i] in ["No loyalty", "Low loyalty"]:
        p[0] += 0.15
    selected = [t for t, prob in zip(switch_triggers, p) if np.random.random() < prob]
    if len(selected) == 0:
        selected = [np.random.choice(switch_triggers)]
    if len(selected) > 2:
        selected = list(np.random.choice(selected, 2, replace=False))
    q12.append("; ".join(selected))
q12 = np.array(q12)

# --- Q13: Price Premium (correlated with income) ---
premium_options = [
    "0% - would not pay more",
    "Up to 10% more",
    "11-25% more",
    "26-50% more",
    "More than 50% more"
]
prem_map_by_income = {
    "Below 5,000": [0.35, 0.35, 0.20, 0.07, 0.03],
    "5,000-10,000": [0.25, 0.35, 0.25, 0.10, 0.05],
    "10,001-20,000": [0.15, 0.30, 0.32, 0.16, 0.07],
    "20,001-35,000": [0.10, 0.25, 0.32, 0.23, 0.10],
    "35,001-50,000": [0.06, 0.18, 0.30, 0.30, 0.16],
    "Above 50,000": [0.04, 0.10, 0.25, 0.35, 0.26],
    "Prefer not to say": [0.15, 0.30, 0.30, 0.18, 0.07],
}
q13 = np.array([np.random.choice(premium_options, p=prem_map_by_income[inc]) for inc in q5])

# --- Q14: Product Categories (multi-select) ---
categories = [
    "Face skincare", "Body care", "Haircare",
    "Fragrance / perfume / oud", "Men's grooming",
    "Oral care", "Colour cosmetics / makeup",
    "Personal hygiene"
]
cat_base = [0.55, 0.48, 0.52, 0.45, 0.25, 0.35, 0.28, 0.42]
q14 = []
for i in range(N):
    p = cat_base.copy()
    if q2[i] == "Male":
        p[4] += 0.35; p[3] += 0.15; p[6] -= 0.15
    if q2[i] == "Female":
        p[0] += 0.15; p[6] += 0.20; p[4] -= 0.15
    if q3[i] == "Emirati":
        p[3] += 0.20
    p = np.clip(p, 0.05, 0.90)
    selected = [c for c, prob in zip(categories, p) if np.random.random() < prob]
    if len(selected) == 0:
        selected = [np.random.choice(categories)]
    q14.append("; ".join(selected))
q14 = np.array(q14)

# --- Q15: Ingredient Importance (6 sub-items, 4-point scale) ---
ingredients = [
    "Halal-certified", "Natural / organic / plant-based",
    "Free from parabens/sulfates/phthalates", "Dermatologist-tested",
    "Vegan and cruelty-free", "Arabian heritage ingredients"
]
scale = ["Not Important", "Somewhat Important", "Very Important", "Essential"]
q15_cols = {}
for ing in ingredients:
    base = [0.15, 0.30, 0.35, 0.20]
    q15_cols[f"Q15_{ing}"] = np.random.choice(scale, N, p=base)
# Adjust halal for Emirati / Arab
for i in range(N):
    if q3[i] in ["Emirati", "Other Arab (Egyptian, Lebanese, Jordanian, etc.)"]:
        if np.random.random() < 0.6:
            q15_cols["Q15_Halal-certified"][i] = np.random.choice(["Very Important", "Essential"])
        if np.random.random() < 0.4:
            q15_cols["Q15_Arabian heritage ingredients"][i] = np.random.choice(["Very Important", "Essential"])
    if q3[i] == "Western Expat (European, American, Australian)":
        if np.random.random() < 0.5:
            q15_cols["Q15_Vegan and cruelty-free"][i] = np.random.choice(["Very Important", "Essential"])

# --- Q16: Product Format (multi-select up to 2) ---
formats = [
    "Lightweight gel / water-based", "Rich cream / butter-based",
    "Serum / oil-based concentrate", "Spray / mist",
    "Multi-use / hybrid products", "Sheet masks / single-use treatments"
]
fmt_probs = [0.35, 0.22, 0.25, 0.18, 0.28, 0.15]
q16 = []
for i in range(N):
    selected = [f for f, p in zip(formats, fmt_probs) if np.random.random() < p]
    if len(selected) == 0:
        selected = [np.random.choice(formats)]
    if len(selected) > 2:
        selected = list(np.random.choice(selected, 2, replace=False))
    q16.append("; ".join(selected))
q16 = np.array(q16)

# --- Q17: Festival Triggers (multi-select) ---
festivals = [
    "Ramadan / Eid al-Fitr", "Eid al-Adha", "Diwali",
    "Christmas / New Year", "Dubai Shopping Festival (DSF / DSS)",
    "Valentine's Day / Mother's Day", "Wedding season / family events",
    "None - consistent spending"
]
fest_base = [0.35, 0.20, 0.18, 0.22, 0.30, 0.15, 0.20, 0.18]
q17 = []
for i in range(N):
    p = fest_base.copy()
    if q3[i] in ["Emirati", "Other Arab (Egyptian, Lebanese, Jordanian, etc.)"]:
        p[0] += 0.25; p[1] += 0.20; p[2] -= 0.10
    if q3[i] == "South Asian (Indian, Pakistani, Bangladeshi, Sri Lankan)":
        p[2] += 0.30; p[0] += 0.10
    if q3[i] == "Western Expat (European, American, Australian)":
        p[3] += 0.25; p[0] -= 0.15
    p = np.clip(p, 0.03, 0.85)
    selected = [f for f, prob in zip(festivals, p) if np.random.random() < prob]
    if len(selected) == 0:
        selected = ["None - consistent spending"]
    q17.append("; ".join(selected))
q17 = np.array(q17)

# --- Q18: Gift Purchases ---
gift_options = [
    "Yes, frequently (5+ times/year)", "Yes, occasionally (2-4 times/year)",
    "Rarely (once a year or less)", "Never"
]
gift_probs = [0.12, 0.35, 0.35, 0.18]
q18 = np.random.choice(gift_options, N, p=gift_probs)

# --- Q19: Cultural Values (multi-select up to 2) ---
values_list = [
    "Religious compliance (halal)",
    "Environmental sustainability",
    "Heritage and tradition",
    "Scientific efficacy",
    "Social image and status",
    "Health and wellness"
]
val_probs = [0.25, 0.20, 0.18, 0.22, 0.18, 0.30]
q19 = []
for i in range(N):
    p = val_probs.copy()
    if q3[i] in ["Emirati", "Other Arab (Egyptian, Lebanese, Jordanian, etc.)"]:
        p[0] += 0.20; p[2] += 0.15
    if q3[i] == "Western Expat (European, American, Australian)":
        p[1] += 0.15; p[3] += 0.10
    p = np.clip(p, 0.05, 0.70)
    selected = [v for v, prob in zip(values_list, p) if np.random.random() < prob]
    if len(selected) == 0:
        selected = [np.random.choice(values_list)]
    if len(selected) > 2:
        selected = list(np.random.choice(selected, 2, replace=False))
    q19.append("; ".join(selected))
q19 = np.array(q19)

# --- Q20: Discovery Methods (multi-select up to 3) ---
discovery = [
    "Instagram", "TikTok", "YouTube",
    "Friends / family", "In-store browsing",
    "Google search / blogs", "Subscription boxes", "Celebrity endorsements"
]
disc_base = [0.42, 0.30, 0.28, 0.35, 0.30, 0.22, 0.08, 0.12]
q20 = []
for i in range(N):
    p = disc_base.copy()
    if q1[i] in ["18-24", "25-34"]:
        p[0] += 0.12; p[1] += 0.15
    if q1[i] in ["45-54", "55+"]:
        p[3] += 0.15; p[4] += 0.15; p[0] -= 0.10; p[1] -= 0.10
    p = np.clip(p, 0.03, 0.80)
    selected = [d for d, prob in zip(discovery, p) if np.random.random() < prob]
    if len(selected) == 0:
        selected = [np.random.choice(discovery)]
    if len(selected) > 3:
        selected = list(np.random.choice(selected, 3, replace=False))
    q20.append("; ".join(selected))
q20 = np.array(q20)

# --- Q21: Influencer Impact (correlated with age) ---
infl_options = [
    "Very strongly", "Moderately", "Slightly", "Not at all"
]
infl_map_by_age = {
    "18-24": [0.30, 0.40, 0.22, 0.08],
    "25-34": [0.22, 0.40, 0.28, 0.10],
    "35-44": [0.10, 0.35, 0.38, 0.17],
    "45-54": [0.06, 0.25, 0.40, 0.29],
    "55+": [0.04, 0.18, 0.38, 0.40],
}
q21 = np.array([np.random.choice(infl_options, p=infl_map_by_age[a]) for a in q1])

# --- Q22: AI Tool ---
ai_options = [
    "Yes, definitely", "Probably", "Unsure", "No"
]
ai_probs = [0.20, 0.35, 0.28, 0.17]
q22 = np.random.choice(ai_options, N, p=ai_probs)

# --- Q23: Discount Type ---
discounts = [
    "Percentage discount (e.g., 20% off)",
    "Buy-one-get-one-free (BOGO)",
    "Free sample / trial size",
    "Loyalty points",
    "Bundle deal",
    "Free shipping",
    "Subscription discount"
]
disc_probs = [0.25, 0.15, 0.18, 0.12, 0.14, 0.10, 0.06]
q23 = np.random.choice(discounts, N, p=disc_probs)

# --- Q24: Loyalty Programme (correlated with age, influencer, AI) ---
loy_prog = ["Very likely", "Likely", "Neutral", "Unlikely", "Very unlikely"]
q24 = []
for i in range(N):
    lp = [0.18, 0.32, 0.28, 0.15, 0.07]
    if q1[i] in ["18-24", "25-34"]: lp[0] += 0.12; lp[1] += 0.08; lp[3] -= 0.08; lp[4] -= 0.05
    if q22[i] in ["Yes, definitely", "Probably"]: lp[0] += 0.10; lp[1] += 0.08; lp[3] -= 0.06
    if q21[i] in ["Very strongly", "Moderately"]: lp[0] += 0.06; lp[1] += 0.06
    lp = np.clip(lp, 0.02, 0.55)
    lp = np.array(lp) / np.array(lp).sum()
    q24.append(np.random.choice(loy_prog, p=lp))
q24 = np.array(q24)

# --- Q25: Brand Interest (target - STRONGLY correlated with features) ---
q25_options = [
    "Very interested", "Interested", "Neutral",
    "Not very interested", "Not interested at all"
]
q25 = []
for i in range(N):
    score = 0.0
    # Strong signals
    if q13[i] == "More than 50% more": score += 0.45
    elif q13[i] == "26-50% more": score += 0.35
    elif q13[i] == "11-25% more": score += 0.18
    elif q13[i] == "Up to 10% more": score += 0.05

    if q11[i] == "No loyalty": score += 0.25
    elif q11[i] == "Low loyalty": score += 0.18
    elif q11[i] == "Neutral": score += 0.05

    if q22[i] == "Yes, definitely": score += 0.22
    elif q22[i] == "Probably": score += 0.12

    if q24[i] == "Very likely": score += 0.22
    elif q24[i] == "Likely": score += 0.12

    if q8[i] == "Above 700": score += 0.20
    elif q8[i] == "401-700": score += 0.15
    elif q8[i] == "201-400": score += 0.08

    if q15_cols["Q15_Halal-certified"][i] == "Essential": score += 0.15
    elif q15_cols["Q15_Halal-certified"][i] == "Very Important": score += 0.08

    if q21[i] == "Very strongly": score += 0.15
    elif q21[i] == "Moderately": score += 0.08

    if q1[i] in ["25-34", "35-44"]: score += 0.06

    if q7[i] == "Weekly": score += 0.12
    elif q7[i] == "Every 2 weeks": score += 0.06

    if q18[i] == "Yes, frequently (5+ times/year)": score += 0.10
    elif q18[i] == "Yes, occasionally (2-4 times/year)": score += 0.05

    # Scoring to probabilities — high score = interested, low score = not interested
    if score >= 1.2:
        probs = [0.55, 0.30, 0.10, 0.03, 0.02]
    elif score >= 0.8:
        probs = [0.35, 0.35, 0.20, 0.07, 0.03]
    elif score >= 0.5:
        probs = [0.15, 0.30, 0.35, 0.15, 0.05]
    elif score >= 0.25:
        probs = [0.08, 0.18, 0.35, 0.28, 0.11]
    else:
        probs = [0.04, 0.10, 0.25, 0.35, 0.26]

    q25.append(np.random.choice(q25_options, p=probs))
q25 = np.array(q25)

# --- Build DataFrame ---
df = pd.DataFrame({
    "Q1_Age_Group": q1,
    "Q2_Gender": q2,
    "Q3_Nationality": q3,
    "Q4_Area": q4,
    "Q5_Income": q5,
    "Q6_Language": q6,
    "Q7_Purchase_Frequency": q7,
    "Q8_Monthly_Spend": q8,
    "Q9_Channels": q9,
    "Q10_Channel_Reason": q10,
    "Q11_Brand_Loyalty": q11,
    "Q12_Switching_Triggers": q12,
    "Q13_Price_Premium": q13,
    "Q14_Product_Categories": q14,
    "Q15_Halal_Certified": q15_cols["Q15_Halal-certified"],
    "Q15_Natural_Organic": q15_cols["Q15_Natural / organic / plant-based"],
    "Q15_Free_From_Chemicals": q15_cols["Q15_Free from parabens/sulfates/phthalates"],
    "Q15_Dermatologist_Tested": q15_cols["Q15_Dermatologist-tested"],
    "Q15_Vegan_Cruelty_Free": q15_cols["Q15_Vegan and cruelty-free"],
    "Q15_Arabian_Heritage": q15_cols["Q15_Arabian heritage ingredients"],
    "Q16_Product_Format": q16,
    "Q17_Festival_Triggers": q17,
    "Q18_Gift_Purchases": q18,
    "Q19_Cultural_Values": q19,
    "Q20_Discovery_Methods": q20,
    "Q21_Influencer_Impact": q21,
    "Q22_AI_Tool_Interest": q22,
    "Q23_Discount_Type": q23,
    "Q24_Loyalty_Programme": q24,
    "Q25_Brand_Interest": q25,
})

# Add some noise / outliers (2% random swaps)
n_noise = int(N * 0.02)
for col in ["Q8_Monthly_Spend", "Q13_Price_Premium", "Q11_Brand_Loyalty"]:
    idx = np.random.choice(N, n_noise, replace=False)
    vals = df[col].unique()
    df.loc[idx, col] = np.random.choice(vals, n_noise)

output_path = os.path.join(os.path.dirname(__file__), "dubai_personal_care_survey.csv")
df.to_csv(output_path, index=False)
print(f"Generated {len(df)} records -> {output_path}")
print(f"Columns: {len(df.columns)}")
print(f"\nQ25 Distribution:\n{df['Q25_Brand_Interest'].value_counts()}")
