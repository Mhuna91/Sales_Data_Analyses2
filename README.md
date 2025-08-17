Sales Data Analysis Dashboard
An interactive dashboard built with Streamlit, Plotly, and Pandas to explore and analyze real-world sales data.
The project uncovers insights into customer behavior, sales performance, and revenue trends.
🔗 Live Demo: [Sales Data Dashboard(https://mhuna91-sales-data-analyses2-app-xsx2ew.streamlit.app/)
Objectives
•	Data Cleaning & Preparation: Handle missing values, format dates, and compute sales totals.
•	Exploratory Data Analysis (EDA):
o	Identify top products and top customers.
o	Analyze sales by day of week and by month.
•	Interactive Visualizations:
o	Line charts, bar charts, scatter plots (Plotly & Matplotlib).
o	Time series decomposition of daily sales.
•	Customer Segmentation: RFM analysis (Recency, Frequency, Monetary) with K-Means clustering.
•	Revenue Forecasting: Predict daily revenue for the next 7 days using XGBoost.
Dataset
The dataset contains retail transaction records with the following fields:
•	OrderID – Unique order identifier
•	Date – Order date
•	CustomerID – Unique customer ID
•	Product – Product name/class
•	Quantity – Number of items purchased
•	Price – Unit price of product (USD)
•	Total – Transaction value (Quantity × Price)
Features in the Dashboard
•	Day of the Week Analysis → Sales quantity & revenue by weekday
•	Top Products → Identify top-selling products
•	Top Customers → Highest value customers
•	Monthly Revenue Trends → Seasonal and long-term patterns
•	Customer Segmentation → RFM-based clustering
•	Forecasting → 7-day sales prediction
Tech Stack
•	Python: pandas, numpy, seaborn, matplotlib, plotly, scikit-learn, statsmodels, xgboost
•	Framework: Streamlit
•	Deployment: Streamlit Cloud
Installation & Usage
Clone the repository and install dependencies:
git clone https://github.com/Mhuna91/Sales_Data_Analyses2.git
cd Sales_Data_Analyses2
pip install -r requirements.txt
Run locally with:
streamlit run app.py

🌐 Deployment
The project is live on Streamlit Cloud:
👉 [Sales Data Dashboard](https://mhuna91-sales-data-analyses2-app-xsx2ew.streamlit.app/)
Author
👤 Munachi Sylvanus Iheanacho
•	🌍 Estate Surveyor & Valuer | Data Science & Machine Learning Enthusiast
•	🔗 [GitHub Profile](https://github.com/Mhuna91)

