
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from datetime import timedelta
import os

# ----------------------------------------
# Load dataset
# ----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset.csv")  # Must be in the same folder as app.py
    df['Date'] = pd.to_datetime(df['Date'])
    df.dropna(subset=['Product', 'Total'], inplace=True)
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    return df

df = load_data()

st.title("Enhanced Sales Data Dashboard")

# Sidebar filters
# ----------------------------------------
st.sidebar.header("Filters")
products = st.sidebar.multiselect("Select Product(s)", df['Product'].unique(), default=df['Product'].unique())
date_range = st.sidebar.date_input("Select Date Range", [df['Date'].min(), df['Date'].max()])

mask = (df['Product'].isin(products)) & (df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))
df_filtered = df[mask]

# ----------------------------------------
# Day of the Week Analysis
# ----------------------------------------
st.header("Customer Patronage by Day of the Week")

sales_by_day = df_filtered.groupby('DayOfWeek')['Quantity'].sum().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])

revenue_by_day = df_filtered.groupby('DayOfWeek')['Total'].sum().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])

# Quantity Plot
st.subheader("Total Quantity Sold by Day")
fig1, ax1 = plt.subplots()
sales_by_day.plot(kind='bar', ax=ax1, color='coral')
ax1.set_ylabel("Quantity Sold")
st.pyplot(fig1)

# Revenue Plot
st.subheader("Total Revenue by Day")
fig2, ax2 = plt.subplots()
revenue_by_day.plot(kind='bar', ax=ax2, color='teal')
ax2.set_ylabel("Revenue ($)")
st.pyplot(fig2)

# ----------------------------------------
# Top Products by Quantity Sold (Plotly)
# ----------------------------------------
st.subheader("Top Products by Quantity Sold")
top_products = df_filtered.groupby('Product')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
fig3 = px.bar(top_products, x='Product', y='Quantity', color='Quantity', title="Top 10 Products")
st.plotly_chart(fig3)

# ----------------------------------------
# Monthly Sales Trend (Plotly)
# ----------------------------------------
st.subheader("Monthly Revenue Trend")
monthly_sales = df_filtered.groupby('Month')['Total'].sum().reset_index()
fig4 = px.line(monthly_sales, x='Month', y='Total', markers=True, title="Revenue Over Time")
st.plotly_chart(fig4)

# ----------------------------------------
# Time Series Decomposition
# ----------------------------------------
st.subheader("Time Series Decomposition (Daily Sales)")
daily_sales = df_filtered.groupby('Date')['Total'].sum()
if len(daily_sales) > 14:  # needs enough data
    result = seasonal_decompose(daily_sales, model='additive', period=7)
    fig5 = result.plot()
    st.pyplot(fig5)

# ----------------------------------------
# Customer Segmentation (RFM + KMeans)
# ----------------------------------------
st.subheader("Customer Segmentation (RFM Analysis)")

snapshot_date = df_filtered['Date'].max() + pd.Timedelta(days=1)
rfm = df_filtered.groupby('CustomerID').agg({
    'Date': lambda x: (snapshot_date - x.max()).days,  # Recency
    'OrderID': 'count',                                # Frequency
    'Total': 'sum'                                     # Monetary
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

fig6 = px.scatter(rfm, x='Recency', y='Monetary', color='Cluster', size='Frequency',
                 hover_name=rfm.index, title='Customer Segments')
st.plotly_chart(fig6)


# # The chart shows customer segmentation using RFM Analysis:
# X-axis: Recency (how recently a customer purchased)
# Y-axis: Monetary (total amount spent)
# Bubble size: Frequency (number of purchases)
# Color: Cluster label

# # Key Insights:
# Lower recency (left) → more recent buyers.
# Higher monetary (top) → more valuable customers.
# Larger bubbles → more frequent purchases.
# You can see clusters such as:
# 
# Valuable Loyal Customers: Low Recency, High Monetary, Large Bubbles
# At-Risk Customers: High Recency, Low Frequency & Monetary
# New or Infrequent Buyers: Low Frequency, Mid Recency

# Forecasting with XGBoost
# ----------------------------------------
st.subheader("Daily Revenue Prediction (Next 7 Days)")

forecast_data = df_filtered.groupby('Date')['Total'].sum().reset_index()
forecast_data = forecast_data.set_index('Date').asfreq('D').fillna(0)
forecast_data['DayOfWeek'] = forecast_data.index.dayofweek
forecast_data['Lag1'] = forecast_data['Total'].shift(1)
forecast_data['Rolling7'] = forecast_data['Total'].rolling(window=7).mean()
forecast_data.dropna(inplace=True)

if len(forecast_data) > 14:
    X = forecast_data[['DayOfWeek', 'Lag1', 'Rolling7']]
    y = forecast_data['Total']

    X_train, y_train = X.iloc[:-7], y.iloc[:-7]
    X_test = X.iloc[-7:]

    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    future_dates = forecast_data.index[-7:] + pd.Timedelta(days=1)
    pred_df = pd.DataFrame({'Date': future_dates, 'PredictedRevenue': y_pred})
    fig7 = px.line(pred_df, x='Date', y='PredictedRevenue', markers=True, title='Predicted Revenue for Next 7 Days')
    st.plotly_chart(fig7)

# ----------------------------------------
# Download Button
# ----------------------------------------
st.sidebar.header("Download Filtered Data")
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df_filtered)
st.sidebar.download_button("Download Filtered Data", csv, "filtered_sales.csv", "text/csv")
