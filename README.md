# 🧴 Dubai Personal Care — Consumer Intelligence Dashboard

**Data-driven go-to-market strategy for a personal care brand launch in Dubai**

A Streamlit dashboard that applies advanced analytics — Classification, K-Means Clustering, Latent Class Analysis, Association Rule Mining, and Regression — to synthetic consumer survey data (5,000 respondents) to design segment-specific marketing strategies for a personal care brand in Dubai.

## 🚀 Live Demo
Deploy on [Streamlit Cloud](https://streamlit.io/cloud) by connecting this GitHub repository.

## 📊 Dashboard Tabs

| Tab | Analytics Type | What It Does |
|-----|---------------|-------------|
| **Descriptive** | Market Snapshot | Demographics, spend, channels, ingredient heatmaps |
| **Diagnostic** | Association Rules + Regression + Crosstabs | Co-purchase patterns, spending drivers, cross-tabulations |
| **Predictive** | Classification + K-Means + LCA | Brand interest prediction, behavioural segmentation, latent classes |
| **Prescriptive** | Strategy per Segment | Product, channel, pricing, and campaign recommendations per cluster |
| **New Customer Predictor** | Upload & Predict | Predict interest and assign segments for new survey data |

## 🛠️ Tech Stack
- **Streamlit** — Dashboard framework
- **Scikit-learn** — Classification (Decision Tree, Random Forest, Gradient Boosted Tree), K-Means, PCA
- **mlxtend** — Apriori association rule mining
- **Plotly** — Interactive visualizations
- **Pandas / NumPy / SciPy** — Data processing

## 📁 File Structure (Flat — no subfolders)
```
app.py                           # Main Streamlit entry point
utils.py                         # Preprocessing utilities and mappings
generate_data.py                 # Synthetic data generation script
tab_descriptive.py               # Descriptive analytics tab
tab_diagnostic.py                # Diagnostic analytics tab (ARM + Regression)
tab_predictive.py                # Predictive analytics tab (Classification + Clustering + LCA)
tab_prescriptive.py              # Prescriptive analytics tab (Strategy cards)
tab_predict_new.py               # New customer prediction tab (Upload & Predict)
dubai_personal_care_survey.csv   # Synthetic dataset (5,000 records)
requirements.txt                 # Python dependencies
.streamlit/config.toml           # Streamlit theme configuration
```

## 🏃 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Deploy on Streamlit Cloud
1. Push all files to a GitHub repository (flat structure, no subfolders)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path to `app.py`
5. Deploy!

## 📝 Survey Design
25-question consumer survey covering demographics, purchase behaviour, channel preferences, price sensitivity, brand loyalty, ingredient consciousness, cultural triggers, social media influence, and brand interest — designed to enable all five analytics techniques.

## 👤 Author
Built as part of an MBA Global Marketing Management capstone project analyzing the Dubai personal care market.
