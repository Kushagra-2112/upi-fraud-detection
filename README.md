<div align="center">

![FraudShield Banner](https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=200&section=header&text=FraudShield&fontSize=60&fontColor=ffffff&fontAlignY=35&desc=Real-Time%20UPI%20Fraud%20Detection&descAlignY=55&descSize=20&descColor=90cdf4&animation=fadeIn)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://upi-fraud-shield.streamlit.app/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

<br>

[![Accuracy](https://img.shields.io/badge/Accuracy-99.86%25-2ecc71?style=flat-square)](https://upi-fraud-shield.streamlit.app/)
[![Transactions](https://img.shields.io/badge/Transactions-6.3M-3498db?style=flat-square)](https://upi-fraud-shield.streamlit.app/)
[![Models](https://img.shields.io/badge/Models-3-9b59b6?style=flat-square)](https://upi-fraud-shield.streamlit.app/)
[![Dataset](https://img.shields.io/badge/Dataset-PaySim-e74c3c?style=flat-square)](https://www.kaggle.com/datasets/ealaxi/paysim1)

<br>

### An AI-powered system that detects fraudulent UPI transactions in real-time

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://upi-fraud-shield.streamlit.app/)

</div>

---

## 🧠 How It Works

Traditional fraud detection relies on simple rules like transaction limits. **FraudShield detects fraud through behavioral patterns** — tracking what happens to balances on both sides of a transaction:

```
✅ Normal Transaction
   Sender:   ₹9,00,000 → ₹0        (balance transferred out)
   Receiver: ₹5,000    → ₹9,05,000  (money arrived as expected)

🚨 Fraud Transaction
   Sender:   ₹9,00,000 → ₹0        (balance transferred out)
   Receiver: ₹5,000    → ₹5,000    (money vanished — never arrived!)
```

> The model asks: *"Money left the sender — but where did it go?"*

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 97.93% | 12.17% | 95.98% | 21.59% |
| ⭐ **Random Forest** | **99.86%** | **69.00%** | **98.23%** | **81.06%** |
| XGBoost | 99.64% | 45.36% | 99.39% | 62.29% |

**Random Forest** is the deployed model — best balance of precision and recall, missing only **29 fraud cases** out of 1,643 in the test set.

---

## ⚙️ App Features

| Page | Description |
|---|---|
| 🔒 **Landing Page** | Fintech-style splash screen |
| ⚡ **Live Simulator** | Simulates random UPI transactions with a real-time risk gauge |
| 🔍 **Check Transaction** | Enter any transaction manually, get an instant fraud prediction |
| 📊 **Model Comparison** | Bar charts and radar chart comparing all 3 models |
| 🎯 **Confusion Matrix** | Visual TP / TN / FP / FN breakdown per model |
| 📈 **Feature Importance** | Which behavioral signals the model relies on most |
| ⚙️ **Threshold Slider** | Tune fraud detection sensitivity live from the sidebar |
| 📋 **Transaction Log** | Running history of all simulated transactions |

---

## 📌 Key Findings

- 🔑 Fraud is detected through **behavioral patterns**, not transaction amount alone
- 🥇 Top fraud signal: `balance_diff` — how much money disappeared from the sender (31.2% importance)
- 🚨 `dest_balance_unchanged` — receiver's balance not changing is a strong fraud indicator
- 🌙 `is_odd_hour` has almost **zero importance** — fraudsters don't only operate at night
- ⚖️ SMOTE was critical — raw data had only **0.129% fraud** (8,213 out of 6,362,620 transactions)

---

## 📈 Feature Importance

```
balance_diff           ████████████████████████  31.2%
oldbalanceOrg          ████████████████           18.7%
amount                 ██████████                 14.0%
dest_balance_unchanged ████████                   11.2%
newbalanceDest         █████                       8.3%
oldbalanceDest         ████                        5.2%
balance_wiped          ██                          3.6%
newbalanceOrig         ██                          3.4%
type_encoded           █                           2.5%
is_large_transaction   █                           1.8%
is_odd_hour            ▏                           0.2%
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| 🐍 Language | Python 3.10+ |
| 📦 Dataset | PaySim Synthetic Dataset — 6.3M transactions |
| 🤖 ML Models | scikit-learn (Logistic Regression, Random Forest), XGBoost |
| ⚖️ Imbalance Handling | imbalanced-learn (SMOTE) |
| 🖥️ Frontend | Streamlit |
| 📊 Visualizations | Plotly |

---

## 🚀 Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Kushagra-2112/upi-fraud-detection
cd upi-fraud-detection

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the PaySim dataset from Kaggle
#    https://www.kaggle.com/datasets/ealaxi/paysim1
#    Place the CSV inside a data/ folder

# 5. Run the training notebook
#    Open 01_eda_and_training.ipynb and run all cells
#    This generates the .pkl model files inside models/

# 6. Launch the Streamlit app
streamlit run app.py
```

---

## 📁 Project Structure

```
upi-fraud-detection/
│
├── 📂 data/                            # PaySim dataset (not in repo — 493 MB)
│
├── 📂 models/                          # Trained model files (generated by notebook)
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl
│
├── 📄 01_eda_and_training.ipynb        # EDA, feature engineering, and model training
├── 📄 app.py                           # Streamlit dashboard
├── 📄 requirements.txt                 # Python dependencies
├── 📄 .gitignore
└── 📄 README.md
```

---

## 🔗 Live Demo

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-upi--fraud--shield.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://upi-fraud-shield.streamlit.app/)

> Try the live app at **[https://upi-fraud-shield.streamlit.app/](https://upi-fraud-shield.streamlit.app/)**

---

## 👤 Author

**Kushagra**

[![GitHub](https://img.shields.io/badge/GitHub-Kushagra--2112-181717?style=for-the-badge&logo=github)](https://github.com/Kushagra-2112)

---

<div align="center">

![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:0f3460,50:16213e,100:1a1a2e&height=100&section=footer)

</div>
