import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix
import os

if 'entered' not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        }
        [data-testid="stHeader"] {background: transparent;}
        [data-testid="stSidebar"] {display: none;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align:center; color:white; font-size:3rem;'>🔒 FraudShield</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#90cdf4; font-size:1.1rem;'>AI-powered real-time UPI transaction protection</p>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)


        st.markdown("<br><br>", unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns([1,2,1])
        with col_b:
            if st.button("🚀  Get Started", use_container_width=True):
                st.session_state.entered = True
                st.rerun()

        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown("""
        <div style='display:flex; justify-content:center; gap:12px; flex-wrap:wrap;'>
            <div style='background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12); border-radius:12px; padding:16px 20px; text-align:center; color:white; width:140px;'>
                <div style='font-size:28px'>⚡</div>
                <div style='font-size:0.9rem; font-weight:600; margin-top:6px;'>Live Simulator</div>
            </div>
            <div style='background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12); border-radius:12px; padding:16px 20px; text-align:center; color:white; width:140px;'>
                <div style='font-size:28px'>🔍</div>
                <div style='font-size:0.9rem; font-weight:600; margin-top:6px;'>Check Transaction</div>
            </div>
            <div style='background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12); border-radius:12px; padding:16px 20px; text-align:center; color:white; width:140px;'>
                <div style='font-size:28px'>📊</div>
                <div style='font-size:0.9rem; font-weight:600; margin-top:6px;'>Model Comparison</div>
            </div>
            <div style='background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12); border-radius:12px; padding:16px 20px; text-align:center; color:white; width:140px;'>
                <div style='font-size:28px'>🎯</div>
                <div style='font-size:0.9rem; font-weight:600; margin-top:6px;'>Confusion Matrix</div>
            </div>
            <div style='background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12); border-radius:12px; padding:16px 20px; text-align:center; color:white; width:140px;'>
                <div style='font-size:28px'>📈</div>
                <div style='font-size:0.9rem; font-weight:600; margin-top:6px;'>Feature Importance</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.stop()

# Page config
st.set_page_config(
    page_title="UPI Fraud Detection",
    page_icon="🔒",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    lr = joblib.load('models/logistic_regression.pkl')
    rf = joblib.load('models/random_forest.pkl')
    xgb = joblib.load('models/xgboost.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return lr, rf, xgb, scaler

lr_model, rf_model, xgb_model, scaler = load_models()

# Session state for transaction log
if 'transaction_log' not in st.session_state:
    st.session_state.transaction_log = []

# Threshold session state
if 'threshold' not in st.session_state:
    st.session_state.threshold = 50

# Sidebar navigation
st.sidebar.title("🔒 UPI Fraud Detection")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "🏠 Dashboard",
    "🔍 Check Transaction",
    "📊 Model Comparison",
    "🎯 Confusion Matrix",
    "📈 Feature Importance"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
threshold = st.sidebar.slider(
    "Fraud Detection Threshold",
    min_value=10,
    max_value=90,
    value=50,
    help="Lower = catch more fraud but more false alarms. Higher = fewer false alarms but may miss fraud."
)
st.sidebar.markdown(f"Current threshold: **{threshold}%**")

# ── PAGE 1: DASHBOARD ──
if page == "🏠 Dashboard":
    st.title("🏠 Real-Time UPI Fraud Detection")
    st.markdown("Simulating live UPI transactions and detecting fraud in real-time.")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", "6,362,620")
    col2.metric("Fraud Cases", "8,213")
    col3.metric("Fraud Rate", "0.129%")
    col4.metric("Best Model Accuracy", "99.86%")

    st.divider()

    # Class imbalance visualization
    st.subheader("⚖️ Class Imbalance in Dataset")
    col1, col2 = st.columns(2)

    with col1:
        fig_imbalance = go.Figure(go.Pie(
            labels=['Genuine', 'Fraud'],
            values=[6354407, 8213],
            hole=0.4,
            marker_colors=['#2ecc71', '#e74c3c']
        ))
        fig_imbalance.update_layout(
            title="Before SMOTE",
            height=250,
            margin=dict(t=40, b=0)
        )
        st.plotly_chart(fig_imbalance, use_container_width=True)

    with col2:
        fig_smote = go.Figure(go.Pie(
            labels=['Genuine', 'Fraud'],
            values=[2209757, 2209757],
            hole=0.4,
            marker_colors=['#2ecc71', '#e74c3c']
        ))
        fig_smote.update_layout(
            title="After SMOTE (balanced)",
            height=250,
            margin=dict(t=40, b=0)
        )
        st.plotly_chart(fig_smote, use_container_width=True)

    st.divider()

    # Live transaction simulator
    st.subheader("⚡ Live Transaction Simulator")
    st.markdown("Click the button to simulate a random UPI transaction.")

    if st.button("🚀 Simulate Transaction", use_container_width=True):
        is_fraud_sim = np.random.choice([0, 1], p=[0.85, 0.15])

        if is_fraud_sim:
            amount = np.random.uniform(50000, 500000)
            oldbalanceOrg = np.random.uniform(amount, amount * 1.1)
            newbalanceOrig = 0.0
            oldbalanceDest = np.random.uniform(0, 1000)
            newbalanceDest = oldbalanceDest
            is_odd_hour = 1
            balance_wiped = 1
        else:
            amount = np.random.uniform(100, 10000)
            oldbalanceOrg = np.random.uniform(amount * 1.5, amount * 3)
            newbalanceOrig = oldbalanceOrg - amount
            oldbalanceDest = np.random.uniform(1000, 50000)
            newbalanceDest = oldbalanceDest + amount
            is_odd_hour = 0
            balance_wiped = 0

        type_encoded = np.random.choice([0, 1])
        balance_diff = oldbalanceOrg - newbalanceOrig
        is_large_transaction = 1 if amount > 274658 else 0
        dest_balance_unchanged = 1 if oldbalanceDest == newbalanceDest else 0

        features = np.array([[
            amount, oldbalanceOrg, newbalanceOrig,
            oldbalanceDest, newbalanceDest, type_encoded,
            is_odd_hour, balance_wiped, balance_diff,
            is_large_transaction, dest_balance_unchanged
        ]])

        scaled = scaler.transform(features)
        risk_score = rf_model.predict_proba(scaled)[0][1] * 100
        label = "FRAUD" if risk_score > threshold else "GENUINE"

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Transaction Details")
            st.markdown(f"**Amount:** ₹{amount:,.2f}")
            st.markdown(f"**Type:** {'TRANSFER' if type_encoded else 'CASH_OUT'}")
            st.markdown(f"**Odd Hour:** {'Yes ⚠️' if is_odd_hour else 'No ✅'}")
            st.markdown(f"**Balance Wiped:** {'Yes ⚠️' if balance_wiped else 'No ✅'}")
            st.markdown(f"**Large Transaction:** {'Yes ⚠️' if is_large_transaction else 'No ✅'}")
            st.markdown(f"**Dest Balance Unchanged:** {'Yes ⚠️' if dest_balance_unchanged else 'No ✅'}")

        with col2:
            st.markdown("### Risk Assessment")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={'text': "Fraud Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if risk_score > threshold else "green"},
                    'steps': [
                        {'range': [0, 40], 'color': "#d4edda"},
                        {'range': [40, 70], 'color': "#fff3cd"},
                        {'range': [70, 100], 'color': "#f8d7da"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

            if label == "FRAUD":
                st.error(f"🚨 FRAUD DETECTED — Risk Score: {risk_score:.1f}%")
            else:
                st.success(f"✅ GENUINE TRANSACTION — Risk Score: {risk_score:.1f}%")

        # Add to transaction log
        st.session_state.transaction_log.append({
            'Type': 'TRANSFER' if type_encoded else 'CASH_OUT',
            'Amount (₹)': f"{amount:,.2f}",
            'Odd Hour': '⚠️ Yes' if is_odd_hour else '✅ No',
            'Balance Wiped': '⚠️ Yes' if balance_wiped else '✅ No',
            'Risk Score': f"{risk_score:.1f}%",
            'Threshold Used': f"{threshold}%",
            'Result': '🚨 FRAUD' if label == 'FRAUD' else '✅ GENUINE'
        })

    # Transaction history log
    if st.session_state.transaction_log:
        st.divider()
        st.subheader("📋 Transaction History")
        log_df = pd.DataFrame(st.session_state.transaction_log)
        st.dataframe(log_df, use_container_width=True)

        fraud_count = sum(1 for t in st.session_state.transaction_log if '🚨' in t['Result'])
        genuine_count = len(st.session_state.transaction_log) - fraud_count

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Simulated", len(st.session_state.transaction_log))
        col2.metric("Fraud Detected", fraud_count)
        col3.metric("Genuine", genuine_count)

        if st.button("🗑️ Clear History"):
            st.session_state.transaction_log = []
            st.rerun()

# ── PAGE 2: CHECK TRANSACTION ──
elif page == "🔍 Check Transaction":
    st.title("🔍 Check a Transaction")
    st.markdown("Enter transaction details manually to check if it is fraudulent.")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Transaction Amount (₹)", min_value=1.0, value=10000.0)
        transaction_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
        oldbalanceOrg = st.number_input("Sender's Balance Before (₹)", min_value=0.0, value=50000.0)
        newbalanceOrig = st.number_input("Sender's Balance After (₹)", min_value=0.0, value=40000.0)
        is_odd_hour = st.selectbox("Transaction at Odd Hour (12am-6am)?", ["No", "Yes"])

    with col2:
        oldbalanceDest = st.number_input("Receiver's Balance Before (₹)", min_value=0.0, value=10000.0)
        newbalanceDest = st.number_input("Receiver's Balance After (₹)", min_value=0.0, value=20000.0)
        model_choice = st.selectbox("Select Model", [
            "Random Forest", "Logistic Regression", "XGBoost"
        ])
        st.markdown(f"**Current Threshold:** {threshold}% *(adjust in sidebar)*")

    if st.button("🔍 Check Fraud", use_container_width=True):
        type_encoded = 1 if transaction_type == "TRANSFER" else 0
        odd_hour = 1 if is_odd_hour == "Yes" else 0
        balance_wiped = 1 if newbalanceOrig == 0 else 0
        balance_diff = oldbalanceOrg - newbalanceOrig
        is_large = 1 if amount > 274658 else 0
        dest_unchanged = 1 if oldbalanceDest == newbalanceDest else 0

        features = np.array([[
            amount, oldbalanceOrg, newbalanceOrig,
            oldbalanceDest, newbalanceDest, type_encoded,
            odd_hour, balance_wiped, balance_diff,
            is_large, dest_unchanged
        ]])

        scaled = scaler.transform(features)

        if model_choice == "Random Forest":
            model = rf_model
        elif model_choice == "Logistic Regression":
            model = lr_model
        else:
            model = xgb_model

        risk_score = model.predict_proba(scaled)[0][1] * 100
        label = "FRAUD" if risk_score > threshold else "GENUINE"

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={'text': "Fraud Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if risk_score > threshold else "green"},
                    'steps': [
                        {'range': [0, 40], 'color': "#d4edda"},
                        {'range': [40, 70], 'color': "#fff3cd"},
                        {'range': [70, 100], 'color': "#f8d7da"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

            if label == "FRAUD":
                st.error(f"🚨 FRAUD DETECTED — Risk Score: {risk_score:.1f}%")
            else:
                st.success(f"✅ GENUINE TRANSACTION — Risk Score: {risk_score:.1f}%")

        with col2:
            st.markdown("### 🔎 Why this prediction?")
            st.markdown("**Fraud signals detected:**")

            signals = []
            if balance_wiped:
                signals.append("🚨 Sender balance wiped to zero")
            if dest_unchanged:
                signals.append("🚨 Receiver balance unchanged")
            if odd_hour:
                signals.append("🚨 Transaction at odd hour")
            if is_large:
                signals.append("⚠️ Large transaction amount")
            if type_encoded:
                signals.append("⚠️ TRANSFER type (higher risk)")

            if signals:
                for s in signals:
                    st.markdown(s)
            else:
                st.markdown("✅ No major fraud signals detected")

            st.markdown("**Key feature values:**")
            st.markdown(f"- Balance difference: ₹{balance_diff:,.2f}")
            st.markdown(f"- Balance wiped: {'Yes' if balance_wiped else 'No'}")
            st.markdown(f"- Dest unchanged: {'Yes' if dest_unchanged else 'No'}")

# ── PAGE 3: MODEL COMPARISON ──
elif page == "📊 Model Comparison":
    st.title("📊 Model Performance Comparison")

    results = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Accuracy': [0.979332, 0.998639, 0.996432],
        'Precision': [0.121654, 0.690038, 0.453611],
        'Recall': [0.959830, 0.982349, 0.993914],
        'F1 Score': [0.215939, 0.810648, 0.622926]
    })

    st.dataframe(results, use_container_width=True)
    st.divider()

    metric = st.selectbox("Select Metric to Visualize", [
        "Accuracy", "Precision", "Recall", "F1 Score"
    ])

    fig = px.bar(
        results,
        x='Model',
        y=metric,
        color='Model',
        title=f'Model Comparison — {metric}',
        text_auto='.3f',
        color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c']
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Radar chart for all metrics
    st.subheader("📡 Radar Chart — All Metrics")
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    fig_radar = go.Figure()
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for i, row in results.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1 Score']],
            theta=categories,
            fill='toself',
            name=row['Model'],
            line_color=colors[i]
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()
    st.subheader("📌 Key Findings")
    st.markdown("""
    - **Random Forest** achieves the best balance of precision and recall
    - **XGBoost** catches the most fraud (highest recall) but with more false alarms
    - **Logistic Regression** is the weakest — high recall but very low precision
    - For a real UPI system, **Random Forest** is the recommended model
    """)

# ── PAGE 4: CONFUSION MATRIX ──
elif page == "🎯 Confusion Matrix":
    st.title("🎯 Confusion Matrix")
    st.markdown("Visual breakdown of correct and incorrect predictions for each model.")

    model_choice = st.selectbox("Select Model", [
        "Random Forest", "Logistic Regression", "XGBoost"
    ])

    # Pre-computed values from training
    cm_data = {
        "Random Forest": {
            "matrix": [[552117, 322], [29, 1614]],
            "tn": 552117, "fp": 322, "fn": 29, "tp": 1614
        },
        "Logistic Regression": {
            "matrix": [[541141, 11298], [66, 1577]],
            "tn": 541141, "fp": 11298, "fn": 66, "tp": 1577
        },
        "XGBoost": {
            "matrix": [[551310, 1129], [10, 1633]],
            "tn": 551310, "fp": 1129, "fn": 10, "tp": 1633
        }
    }

    cm = cm_data[model_choice]

    fig_cm = px.imshow(
        cm["matrix"],
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Genuine', 'Fraud'],
        y=['Genuine', 'Fraud'],
        color_continuous_scale='RdYlGn',
        text_auto=True
    )
    fig_cm.update_layout(
        title=f"Confusion Matrix — {model_choice}",
        height=400
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("True Negative ✅", f"{cm['tn']:,}", help="Correctly identified genuine")
    col2.metric("False Positive ⚠️", f"{cm['fp']:,}", help="Genuine flagged as fraud")
    col3.metric("False Negative 🚨", f"{cm['fn']:,}", help="Fraud missed by model")
    col4.metric("True Positive ✅", f"{cm['tp']:,}", help="Correctly identified fraud")

    st.divider()
    st.markdown("""
    **What these mean:**
    - **True Negative** — genuine transactions correctly approved ✅
    - **False Positive** — genuine transactions wrongly flagged (customer inconvenience) ⚠️
    - **False Negative** — fraud missed by the model (most dangerous) 🚨
    - **True Positive** — fraud correctly caught ✅

    **Random Forest minimizes False Negatives** — it misses only 29 fraud cases out of 1,643.
    """)

# ── PAGE 5: FEATURE IMPORTANCE ──
elif page == "📈 Feature Importance":
    st.title("📈 Feature Importance")
    st.markdown("Which features matter most for fraud detection?")

    feature_data = pd.DataFrame({
        'Feature': [
            'balance_diff', 'oldbalanceOrg', 'amount',
            'dest_balance_unchanged', 'newbalanceDest',
            'oldbalanceDest', 'balance_wiped', 'newbalanceOrig',
            'type_encoded', 'is_large_transaction', 'is_odd_hour'
        ],
        'Importance': [
            0.311532, 0.187083, 0.139829,
            0.111875, 0.083103,
            0.052005, 0.036237, 0.033728,
            0.025085, 0.017679, 0.001844
        ]
    }).sort_values('Importance', ascending=True)

    fig = px.bar(
        feature_data,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance — Random Forest',
        color='Importance',
        color_continuous_scale='viridis',
        text_auto='.3f'
    )
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📌 What this means")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Top 3 features:**
        1. `balance_diff` (31.2%) — how much money disappeared from sender
        2. `oldbalanceOrg` (18.7%) — how much sender had originally
        3. `amount` (14.0%) — size of the transaction
        """)
    with col2:
        st.markdown("""
        **Key insight:**
        - Amount alone is NOT the top signal
        - The **combination** of balance changes is what catches fraud
        - `is_odd_hour` has almost no importance — fraudsters don't only operate at night
        """)