import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("data/superstore_sales.csv", encoding="latin1")

# Show first rows
print(data.head())

# Sales by Region
sales_region = data.groupby("Region")["Sales"].sum()

print("\nSales by Region:")
print(sales_region)

# Plot graph
sales_region.plot(kind="bar")

plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")

plt.show()
# Sales by Category
sales_category = data.groupby("Category")["Sales"].sum()

print("\nSales by Category:")
print(sales_category)

# Plot graph
sales_category.plot(kind="bar")

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")

plt.show()
# Convert Order Date to datetime
data["Order Date"] = pd.to_datetime(data["Order Date"])

# Create Month-Year column
data["Month"] = data["Order Date"].dt.to_period("M")

# Calculate monthly sales
monthly_sales = data.groupby("Month")["Sales"].sum()
monthly_sales_df = monthly_sales.reset_index()

# Moving Average (window = 3 months)
monthly_sales_df['Moving_Avg'] = monthly_sales_df['Sales'].rolling(window=3).mean()

print("\nMonthly Sales:")
print(monthly_sales)

plt.figure(figsize=(12,6))

plt.plot(monthly_sales_df['Month'].astype(str), monthly_sales_df['Sales'], label='Actual Sales')
plt.plot(monthly_sales_df['Month'].astype(str), monthly_sales_df['Moving_Avg'], label='Moving Average')

plt.xticks(rotation=45)
plt.title('Monthly Sales Trend with Moving Average')
plt.xlabel('Month')
plt.ylabel('Sales')

plt.legend()
plt.show()

# Moving Average (window = 3 months)

monthly_sales_df['Month_Num'] = range(len(monthly_sales_df))
X = monthly_sales_df[['Month_Num']]
y = monthly_sales_df['Sales']
model = LinearRegression()
model.fit(X, y)
future_months = [[i] for i in range(len(monthly_sales_df), len(monthly_sales_df)+6)]
predictions = model.predict(future_months)
print("\nFuture Sales Predictions:")
print(predictions)
future_x = list(range(len(monthly_sales_df), len(monthly_sales_df)+6))

plt.figure(figsize=(12,6))

plt.plot(monthly_sales_df['Month_Num'], monthly_sales_df['Sales'], label='Actual Sales')
plt.plot(future_x, predictions, linestyle='dashed', label='Forecast')

plt.title('Sales Forecasting')
plt.xlabel('Time')
plt.ylabel('Sales')

plt.legend()
plt.show()
