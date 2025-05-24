# Sales Dashboard - Data Analysis

# Essential Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sqlalchemy import create_engine, text
import dash
from dash import dcc, html, Input, Output

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load sales data from CSV and perform initial cleaning steps

    Args:
        filepath: Path to the CSV file

    Returns:
        Cleaned pandas DataFrame ready for analysis
    """
    print(f"Loading data from: {filepath}")

    # Load dataset with error handling
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty: {filepath}")

    # Display initial dataset info
    print("\nInitial Dataset Info:")
    df.info()
    print("\nFirst few rows:")
    print(df.head())

    # Data cleaning steps
    initial_rows = len(df)

    # 1. Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # 2. Check and handle missing values
    missing = df.isnull().sum()
    if missing.any():
        print("\nMissing values found:")
        print(missing[missing > 0])
        # Handle missing values based on column type
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        df = df.fillna('Unknown')  # Fill remaining with 'Unknown'

    # 3. Remove duplicates
    df = df.drop_duplicates()
    dropped_rows = initial_rows - len(df)
    if dropped_rows:
        print(f"\nRemoved {dropped_rows} duplicate rows")

    # 4. Generate summary statistics
    print("\nSummary Statistics:")
    print(df.describe(include='all'))

    # 5. Validate data types
    try:
        # Convert date column to datetime if exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        print(f"Warning: Could not convert date column - {str(e)}")

    print("\nData cleaning completed successfully!")
    return df

# Load and clean the dataset
data = load_and_clean_data("sales_data.csv")

#### 1️⃣ Sales Trends (Daily, Monthly, Yearly Growth)
# Convert 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Daily Sales Trend
daily_sales = data.groupby('date')['revenue'].sum().reset_index()

new_data = data.copy()
# Monthly Sales Trend
new_data['month_year'] = new_data['date'].dt.to_period('M')
monthly_sales = new_data.groupby('month_year')['revenue'].sum().reset_index()

# Yearly Sales Trend
yearly_sales = new_data.groupby('year')['revenue'].sum().reset_index()

# Lineplot for daily sales
plt.figure(figsize = (15,6))
fig_daily = sns.lineplot(x='date', y='revenue', data = daily_sales, label ="Daily Sales Trend")
plt.xticks(rotation = 45)
plt.title("Sales Trend Over Time")
plt.show()

# Lineplot for monthly sales
monthly_sales['month_year'] = monthly_sales['month_year'].astype(str)

plt.figure(figsize=(15,10))
fig_monthly = plt.plot(monthly_sales['month_year'],monthly_sales['revenue'], marker='o')

plt.title('Monthly Sales Trend', fontsize=14)
plt.xlabel('Month-Year', fontsize=12)
plt.ylabel('Revenue', fontsize=12)
plt.xticks(rotation = 90)

plt.tight_layout()
plt.show()

# Lineplot for yearly sales
plt.figure(figsize = (10,6))
# Customize bar color
fig_yearly = plt.bar(yearly_sales['year'], yearly_sales['revenue'], color='teal', edgecolor='black', linewidth=1.9)

plt.title('Yearly Sales Trend', fontsize=16, fontweight='bold', color='darkblue')
plt.xlabel('Year', fontsize=14, fontweight='bold')
plt.ylabel('Revenue', fontsize=14, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

#### 2️⃣ Top Products by Revenue & Quantity
# Top 10 Products by Revenue
top_products_revenue = data.groupby('product')['revenue'].sum().nlargest(10).reset_index()

# Top 10 Products by Quantity
top_products_quantity = data.groupby('product')['order_quantity'].sum().nlargest(10).reset_index()

# Plot
fig, axes = plt.subplots(1,2, figsize = (14, 6))

fig_revenue = sns.barplot(x='revenue', y='product', data=top_products_revenue, ax=axes[0], palette="viridis")
axes[0].set_title('Top 10 Products by Revenue', fontsize=16, fontweight='bold', color='darkblue')
axes[0].set_xlabel('Revenue', fontsize=14)
axes[0].set_ylabel('Product', fontsize=14)

fig_quantity = sns.barplot(x='product', y='order_quantity', data=top_products_quantity, ax=axes[1], palette="coolwarm")
axes[1].set_title('Top 10 Products by Quantity', fontsize=16, fontweight='bold', color='darkblue')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation = 90, ha="right")
axes[1].set_xlabel('Product', fontsize=14)
axes[1].set_ylabel('Quantity Sold', fontsize=14)

for ax in axes:
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=10)

# Adjust space between subplots
plt.subplots_adjust(wspace=0.3)  # Increase the value for more space between subplots

#### 3️⃣ Customer Segmentation (Age Groups & Location-Wise Sales)
# Sales by Age Groups
age_sales = data.groupby('age_group')['revenue'].sum().reset_index()

# Sales by Location (Country-Wise)
location_sales = data.groupby('country')['revenue'].sum().reset_index()

# Pie Chart for Age Groups
plt.figure(figsize=(12,8))
fig_age = plt.pie(age_sales['revenue'],labels=age_sales['age_group'],autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title('Revenue Distribution by Age Group')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
plt.show()

#Pie Chart for Location
plt.figure(figsize=(12,8))
fig_location = plt.pie(location_sales['revenue'], labels=location_sales['country'], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title('Revenue Distribution by Location')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
plt.show()

#### 4️⃣ Seasonality Analysis (Peak Sales Periods)
# Average Revenue by Month
seasonality = data.groupby('month')['revenue'].sum().reset_index()

# Plot Seasonality
plt.figure(figsize=(12,5))
fig_seasonality = sns.lineplot(x='month', y='revenue', data=seasonality, marker='o')
plt.xticks(range(1,13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.title("Seasonality: Sales Trend by Month")
plt.show()

##  Integrate Findings into Dash
# Create a list of months in the correct order
months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("📊 Sales Dashboard", style = {'textAlign':'center'}),

    # Filters: Month and Year
    html.Div([
        html.Label("Select Month"),
        dcc.Dropdown(
            id='month-filter',
            options=[{'label': i, 'value': i} for i in months_order],
            value=None,  # Default value (None means no filter applied)
            placeholder="Select a Month"
        ),
        html.Label("Select Year"),
        dcc.Dropdown(
            id='year-filter',
            options=[{'label':str(year), 'value':year} for year in sorted(data['year'].unique())],
            value=None,  # Default value (None means no filter applied)
            placeholder='Select Year',
        )
    ], style={'padding':'20px'}),

    #Sales Trends
    html.Div([
        html.H2("Sales Trends"),
        dcc.Graph(id="sales-trend-daily"),
        dcc.Graph(id="sales-trend-monthly"),
        dcc.Graph(id="sales-trend-yearly")

    ], style={'padding': '20px'}),

    # Top Products
    html.Div([
        html.H2("Top Products by Revenue & Quantity"),
        dcc.Graph(id="top-products-by-revenue"),
        dcc.Graph(id="top-products-by-quantity")
    ], style={'padding': '20px'}),

    # Customer Segmentation
    html.Div([
        html.H2("Customer Segmentation"),
        dcc.Graph(id="customer-segmentation-by-age"),
        dcc.Graph(id="customer-segmentation-by-location")
    ], style={"padding":'20px'}),

    # Seasonality
    html.Div([
        html.H2("Seasonality Analysis"),
        dcc.Graph(id="seasonality-analysis")
    ], style = {"padding":'20px'})
])

# Sales Trend - Daily, Monthly, Yearly
@app.callback(
    Output('sales-trend-daily', 'figure'),
    Output('sales-trend-monthly', 'figure'),
    Output('sales-trend-yearly', 'figure'),
    Input('month-filter', 'value'),
    Input('year-filter', 'value')
)

def update_sales_trends(selected_month, selected_year):
    # Filter data based on selected month and year
    filtered_data = data.copy()
    if selected_month:
        filtered_data = filtered_data[filtered_data['month'] == selected_month]
    if selected_year:
        filtered_data = filtered_data[filtered_data['year'] == selected_year]

    # Daily Sales
    daily_sales = filtered_data.groupby('date')['revenue'].sum().reset_index()
    fig_daily = px.line(daily_sales, x='date', y='revenue', title='Daily Sales Trend')

    # Monthly Sales (only show if month is selected, or both month and year are selected)
    if selected_month and not selected_year:
        filtered_data['month_year'] = filtered_data['date'].dt.to_period('M')
        monthly_sales = filtered_data.groupby('month_year')['revenue'].sum().reset_index()
        monthly_sales['month_year'] = monthly_sales['month_year'].astype(str)
        fig_monthly = px.line(monthly_sales, x='month_year', y='revenue', title='Monthly Sales Trend')
    elif selected_year and selected_month:
        # If only year is selected, we show yearly graph
        fig_monthly = {}
    else:
        # Default monthly trend (if no filters are selected)
        filtered_data['month_year'] = filtered_data['date'].dt.to_period('M')
        monthly_sales = filtered_data.groupby('month_year')['revenue'].sum().reset_index()
        monthly_sales['month_year'] = monthly_sales['month_year'].astype(str)
        fig_monthly = px.line(monthly_sales, x='month_year', y='revenue', title='Monthly Sales Trend')

    # Yearly Sales (only show if year is selected)
    if not selected_year and not selected_month:
        yearly_sales = filtered_data.groupby('year')['revenue'].sum().reset_index()
        fig_yearly = px.bar(yearly_sales, x='year', y='revenue', title='Yearly Sales Trend')
    if selected_month and not selected_year:
        yearly_sales = filtered_data.groupby('year')['revenue'].sum().reset_index()
        fig_yearly = px.bar(yearly_sales, x='year', y='revenue', title='Yearly Sales Trend')
    else:
        fig_yearly = {}

    return fig_daily, fig_monthly, fig_yearly

# Top Products by Revenue & Quantity
@app.callback(
    Output('top-products-by-revenue', 'figure'),
    Output('top-products-by-quantity', 'figure'),
    Input('month-filter', 'value'),
    Input('year-filter', 'value')
)

def update_top_products(selected_month, selected_year):

    # Filter data based on selected month and year
    filtered_data = data.copy()
    if selected_month:
        filtered_data = filtered_data[filtered_data['month'] == selected_month]
    if selected_year:
        filtered_data = filtered_data[filtered_data['year'] == selected_year]

    # Top Products by Revenue
    top_products_revenue = filtered_data.groupby('product')['revenue'].sum().nlargest(10).reset_index()
    fig_revenue = px.bar(top_products_revenue, x='revenue', y='product', orientation='h', title='Top 10 Products by Revenue')

    # Top Products by Quantity
    top_products_quantity = filtered_data.groupby('product')['order_quantity'].sum().nlargest(10).reset_index()
    fig_quantity = px.bar(top_products_quantity, x='order_quantity', y='product', orientation='h', title='Top 10 Products by Quantity')

    return fig_revenue, fig_quantity

# Customer Segmentation (Age and Location)
@app.callback(
    Output('customer-segmentation-by-age', 'figure'),
    Output('customer-segmentation-by-location', 'figure'),
    Input('month-filter', 'value'),
    Input('year-filter', 'value')
)

def update_customer_segmentation(selected_month, selected_year):

    # Filter data based on selected month and year
    filtered_data = data.copy()
    if selected_month:
        filtered_data = filtered_data[filtered_data['month'] == selected_month]
    if selected_year:
        filtered_data = filtered_data[filtered_data['year'] == selected_year]

    # Sales by Age Groups
    age_sales = filtered_data.groupby('age_group')['revenue'].sum().reset_index()
    fig_age = px.pie(age_sales, names='age_group', values='revenue', title='Revenue Distribution by Age Group')

    # Sales by Location (Country-Wise)
    location_sales = filtered_data.groupby('country')['revenue'].sum().reset_index()
    fig_location = px.pie(location_sales, names='country', values='revenue', title='Revenue Distribution by Location')

    return fig_age, fig_location

# Seasonality Analysis
@app.callback(
    Output('seasonality-analysis', 'figure'),
    Input('month-filter', 'value'),
    Input('year-filter', 'value')
)

def update_seasonality(selected_month, selected_year):

     # Filter data based on selected month and year
    filtered_data = data.copy()
    if selected_month:
        filtered_data = filtered_data[filtered_data['month'] == selected_month]
    if selected_year:
        filtered_data = filtered_data[filtered_data['year'] == selected_year]

    seasonality = filtered_data.groupby('month')['revenue'].sum().reset_index()
    fig_seasonality = px.line(seasonality, x='month', y='revenue', title='Seasonality: Sales Trend by Month')
    return fig_seasonality

if __name__ == '__main__':
    app.run_server(debug=True)
# Define your PostgreSQL credentials
DB_NAME = 'sales_dashboard'
DB_USER = 'rkuku'
DB_PASSWORD = 'rkuku'
DB_HOST = 'localhost'
DB_PORT = '5432'
table_name = 'sales_data'
# Connect to PostgreSQL using SQLAlchemy
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
# Define the table creation query
create_table_query = """
CREATE TABLE IF NOT EXISTS sales_data (
    date DATE,
    day INT,
    month VARCHAR(20),
    year INT,
    customer_age INT,
    age_group VARCHAR(50),
    customer_gender VARCHAR(10),
    country VARCHAR(50),
    state VARCHAR(50),
    product_category VARCHAR(50),
    sub_category VARCHAR(50),
    product VARCHAR(100),
    order_quantity INT,
    unit_cost FLOAT,
    unit_price FLOAT,
    profit FLOAT,
    cost FLOAT,
    revenue FLOAT
);
"""
# Execute the query to create the table
with engine.connect() as conn:
    conn.execute(text(create_table_query))
    conn.commit()

print("Table 'sales_data' created successfully.")
# Store the DataFrame in PostgreSQL
data.to_sql(table_name, engine, if_exists='append', index=False)

print("Data inserted successfully!")