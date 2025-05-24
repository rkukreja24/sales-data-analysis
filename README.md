# 🛒 Sales Data Analysis Project

This project explores and analyzes a retail sales dataset to extract actionable business insights. Alongside the technical implementation in Python, PostgreSQL, Dash, and Power BI, I used [Monday.com](https://monday.com) for structured project management.

---

## 📂 Dataset Source

- **Title:** Sales Data
- **Author:** Jehanzaib Bhatti
- **Link:** [Kaggle - Sales Data](https://www.kaggle.com/datasets/jehanzaibbhatti/sales-data)
- **Description:** Contains retail transaction details including customer demographics, product information, profit, and revenue metrics.

---

## 🎯 Objective

- Load, clean, and analyze sales data
- Store structured data in a PostgreSQL database
- Build dynamic dashboards using Dash and Power BI to visualize KPIs
- Derive business insights and seasonal trends

---

## 🧰 Tools & Technologies

- **Python** – Data cleaning, transformation, and PostgreSQL integration
- **Pandas, NumPy** – Data manipulation
- **Matplotlib, Seaborn, Plotly** – Visualizations
- **SQLAlchemy** – PostgreSQL connection
- **PostgreSQL** – Relational database storage
- **Dash (Plotly)** – Interactive web dashboard
- **Power BI** – Business reporting dashboard
- **Monday.com** – Project planning & task tracking

---

## 🗂️ Project Structure

```
sales-data-analysis/
│── visualization/
│ ├── PowerBI.png
│ ├── dash_dashboard.png
├── sales_data.csv
├── main.ipynb
├── previousCode.py
├── SalesDashboard.pbit
├── README.md
└── Project_Roadmap.docx
```

---

---

## 📊 Dash Dashboard (Python)

I built a **live, interactive web dashboard** using **Plotly Dash** to visualize key performance indicators directly from the PostgreSQL database.

**Key Features:**

- 📈 Monthly revenue trends
- 🏆 Top product categories
- 👥 Customer segmentation by age and gender
- 🗺️ Regional sales performance

📷 **Screenshot – Dash Dashboard**
![Dash Dashboard](visualization/dash_dashboard.mp4)

---

## 📊 Power BI Dashboard

After storing the data in **PostgreSQL**, I connected **Power BI** to create a comprehensive and interactive dashboard for business stakeholders.

**Key Features:**

- 🎛️ Filters by date, country, and product category
- 📉 Sales vs Profit visual breakdown
- 🔢 Dynamic KPI cards and time series analysis
- 👤 Customer demographics insights

📷 **Screenshot – Power BI Dashboard**
![Power BI Dashboard](visualization/PowerBI.png)

---

## 📈 Key Insights

- **Top Products:** Identified best-selling products based on revenue
- **Seasonality:** Clear seasonal trends in product demand and sales
- **Customer Insights:** Higher profit margins in specific age groups and regions
- **Profit Drivers:** Certain sub-categories deliver consistently high profit-to-cost ratios


## ✅ Next Steps

- Implement advanced time-series forecasting (e.g., ARIMA, Prophet)
- Incorporate external events (promotions, holidays)
- Extend dashboard with real-time data or deploy as a web app

---

## 👩‍💻 Author

**Ritu Kukreja**
📧 [rvkukreja24@gmail.com](mailto:rvkukreja24@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/ds-rvk)
💻 [GitHub](https://github.com/rkukreja24)
---

## 📜 License

This project is for educational and demonstrational purposes. Please refer to the dataset’s license on Kaggle for data usage terms.
