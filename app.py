import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px

# Set up the page
st.set_page_config(page_title="Finance ML App", layout="wide")
st.title("💸 Finance ML App with Kragle + Yahoo + ML")

# ---- SIDEBAR ----
st.sidebar.title("💼 Data Options")

# Kragle Upload
uploaded_file = st.sidebar.file_uploader("📁 Upload Kragle CSV file", type=["csv"])

# Yahoo Finance Fetch
st.sidebar.subheader("📈 Fetch Real-Time Stock Data")
stock_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")

# ML Model selection
st.sidebar.subheader("🤖 Choose Your ML Model")
model_option = st.sidebar.selectbox("Select Model", ["Linear Regression"])

# ---- LOAD KRAGLE DATA ----
if uploaded_file is not None:
    df_kragle = pd.read_csv(uploaded_file)
    st.success("✅ Kragle dataset uploaded!")
    st.write("### 📊 Preview of Uploaded Kragle Data")
    st.dataframe(df_kragle.head())

    # --- Clean Data ---
    df_kragle.dropna(inplace=True)

    # --- Feature Engineering (Kragle) ---
    try:
        X = df_kragle[['Volume']]
        y = df_kragle['Price']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### 🧠 Kragle Linear Regression Results")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R²): {r2:.2f}")

        # Plot
        fig, ax = plt.subplots()
        ax.plot(y_test.index, y_test, label="Actual", marker="o", color='blue')
        ax.plot(y_test.index, y_pred, label="Predicted", marker="x", color='red')
        ax.set_title("Kragle: Actual vs Predicted Prices")
        ax.legend()
        st.pyplot(fig)

        # Plotly
        fig_plot = px.line(x=y_test.index, y=[y_test, y_pred], labels={'x': 'Index', 'y': 'Price'},
                           title='📈 Kragle: Predicted vs Actual Prices')
        st.plotly_chart(fig_plot)

    except Exception as e:
        st.error(f"❌ Error processing Kragle dataset: {e}")

# ---- YAHOO FINANCE DATA ----
if stock_symbol:
    stock_data = yf.download(stock_symbol, period="5d", interval="1d")
    if not stock_data.empty:
        st.write(f"### 📉 Real-Time Data for {stock_symbol}")
        st.dataframe(stock_data)

        # Basic line plot of Close price
        fig = px.line(stock_data, x=stock_data.index, y='Close', title=f'{stock_symbol} Closing Prices')
        st.plotly_chart(fig)
    else:
        st.warning("⚠️ Could not fetch stock data.")
else:
    st.info("👈 Enter a ticker symbol to get real-time stock data.")



