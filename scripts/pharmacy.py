import pandas as pd
import os
import logging
# Define the directory containing your datasets
data = r"C:\Users\Hasan\Desktop\week4"

# File names
files = ["store.csv", "test.csv", "train.csv"]

# Read and log datasets
for file in files:
    try:
        file_path = os.path.join(data, file)
        df = pd.read_csv(file_path)  # Read the CSV file
        
        # Log the file name and its contents (first few rows)
        logging.info(f"Contents of {file}:\n{df.head().to_string(index=False)}")
        print(f"{file} processed and logged successfully.")
        # Print the file's head to the console
        print(f"\nContents of {file}:")
        print(df.head())
    except Exception as e:
        # Log any errors that occur during processing
        logging.error(f"Error processing {file}: {e}")
        print(f"Error processing {file}. Check log for details.")
        # Define the file path
data = r"C:\Users\Hasan\Desktop\week4"
store_file = os.path.join(data, "store.csv")

try:
    # Load the dataset
    store_df = pd.read_csv(store_file)

    # Check for missing values before filling
    print("Missing values before filling:")
    print(store_df[['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']].isnull().sum())

    # Fill missing values
    store_df['Promo2SinceWeek'] = store_df['Promo2SinceWeek'].fillna(store_df['Promo2SinceWeek'].mean())  # Fill with mean
    store_df['Promo2SinceYear'] = store_df['Promo2SinceYear'].fillna(store_df['Promo2SinceYear'].mean())  # Fill with mean
    store_df['PromoInterval'] = store_df['PromoInterval'].fillna(store_df['PromoInterval'].mode()[0])  # Fill with mode

    # Check for missing values after filling
    print("\nMissing values after filling:")
    print(store_df[['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']].isnull().sum())

    # Save the cleaned dataset (optional)
    cleaned_file = os.path.join(data, "store_cleaned.csv")
    store_df.to_csv(cleaned_file, index=False)
    print(f"Cleaned data saved to {cleaned_file}")
except Exception as e:
    print(f"Error processing store.csv: {e}")

# Set up logging
logging.basicConfig(
    filename="promo_comparison.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# File paths
data = r"C:\Users\Hasan\Desktop\week4"
train_file = os.path.join(data, "train.csv")
test_file = os.path.join(data, "test.csv")

try:
    # Load datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Check if 'Promo' column exists in both datasets
    if 'Promo' in train_df.columns and 'Promo' in test_df.columns:
        # Calculate the distribution of 'Promo' for train and test datasets
        train_promo_dist = train_df['Promo'].value_counts(normalize=True) * 100
        test_promo_dist = test_df['Promo'].value_counts(normalize=True) * 100

        # Log the distributions
        logging.info("Promotion distribution in train.csv:\n" + train_promo_dist.to_string())
        logging.info("Promotion distribution in test.csv:\n" + test_promo_dist.to_string())

        # Print the distributions to the console
        print("\nPromotion distribution in train.csv (%):")
        print(train_promo_dist)

        print("\nPromotion distribution in test.csv (%):")
        print(test_promo_dist)

        # Compare the distributions visually
        plt.figure(figsize=(10, 5))
        plt.bar(train_promo_dist.index - 0.2, train_promo_dist.values, width=0.4, label='Train', color='blue')
        plt.bar(test_promo_dist.index + 0.2, test_promo_dist.values, width=0.4, label='Test', color='purple')
        plt.title("Promotion Distribution Comparison")
        plt.xlabel("Promo")
        plt.ylabel("Percentage")
        plt.xticks([0, 1], labels=["Not in Promo (0)", "In Promo (1)"])
        plt.legend()
        plt.show()
    else:
        if 'Promo' not in train_df.columns:
            logging.warning("'Promo' column not found in train.csv.")
            print("'Promo' column not found in train.csv.")
        if 'Promo' not in test_df.columns:
            logging.warning("'Promo' column not found in test.csv.")
            print("'Promo' column not found in test.csv.")
except Exception as e:
    logging.error(f"Error comparing Promo distribution: {e}")
    print(f"Error comparing Promo distribution. Check log for details.")

# Set up logging
logging.basicConfig(
    filename="sales_behavior.log",  # Log file name
    level=logging.INFO,             # Log level
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# File path for train.csv
data = r"C:\Users\Hasan\Desktop\week4"
train_file = os.path.join(data, "train.csv")

try:
    # Load the dataset
    logging.info("Loading train.csv...")
    df = pd.read_csv(train_file)
    logging.info(f"train.csv loaded successfully. Shape: {df.shape}")

    # Convert 'Date' to datetime format for easier time-based calculations
    logging.info("Converting 'Date' column to datetime format...")
    df['Date'] = pd.to_datetime(df['Date'])

    # Ensure 'StateHoliday' is treated as a string for clarity
    logging.info("Ensuring 'StateHoliday' is treated as a string...")
    df['StateHoliday'] = df['StateHoliday'].astype(str)

    # Sort data by Store and Date for logical grouping
    logging.info("Sorting data by Store and Date...")
    df = df.sort_values(by=['Store', 'Date']).reset_index(drop=True)

    # Define a function to label sales as "Before", "During", or "After" holidays
    def label_holiday_period(row, holidays):
        if row['StateHoliday'] != '0':
            return 'During'
        elif (row['Date'] - pd.Timedelta(days=7)) in holidays.values:
            return 'Before'
        elif (row['Date'] + pd.Timedelta(days=7)) in holidays.values:
            return 'After'
        else:
            return 'Regular'

    # Identify unique holiday dates
    logging.info("Identifying unique holiday dates...")
    holiday_dates = df[df['StateHoliday'] != '0']['Date']

    # Add a new column to categorize sales periods
    logging.info("Categorizing sales periods into 'Before', 'During', and 'After'...")
    df['SalesPeriod'] = df.apply(lambda x: label_holiday_period(x, holiday_dates), axis=1)

    # Group by SalesPeriod and calculate average and total sales
    logging.info("Calculating sales statistics for each sales period...")
    sales_analysis = df.groupby('SalesPeriod')['Sales'].agg(['mean', 'sum']).reset_index()
    logging.info("Sales analysis completed.")
    logging.info(f"\n{sales_analysis}")

    # Print sales analysis to the console
    print("\nSales Analysis by Period:")
    print(sales_analysis)

    # Visualize sales behavior across periods
    logging.info("Visualizing total sales by sales period...")
    plt.figure(figsize=(10, 6))
    plt.bar(sales_analysis['SalesPeriod'], sales_analysis['sum'], color=['blue', 'orange', 'green', 'gray'])
    plt.title("Total Sales by Sales Period")
    plt.xlabel("Sales Period")
    plt.ylabel("Total Sales")
    plt.savefig("total_sales_by_period.png")  # Save the plot as a file
    plt.show()
    logging.info("Total sales bar chart saved as 'total_sales_by_period.png'.")

    logging.info("Visualizing average sales by sales period...")
    plt.figure(figsize=(10, 6))
    plt.plot(sales_analysis['SalesPeriod'], sales_analysis['mean'], marker='o', color='purple')
    plt.title("Average Sales by Sales Period")
    plt.xlabel("Sales Period")
    plt.ylabel("Average Sales")
    plt.grid()
    plt.savefig("average_sales_by_period.png")  # Save the plot as a file
    plt.show()
    logging.info("Average sales line plot saved as 'average_sales_by_period.png'.")

except Exception as e:
    logging.error(f"An error occurred: {e}")
    print(f"An error occurred. Check the log file for details.")

# Set up logging
logging.basicConfig(
    filename="seasonal_behavior.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# File path for train.csv
data = r"C:\Users\Hasan\Desktop\week4"
train_file = os.path.join(data, "train.csv")

try:
    # Load the dataset
    logging.info("Loading train.csv...")
    df = pd.read_csv(train_file)
    logging.info(f"train.csv loaded successfully. Shape: {df.shape}")

    # Preprocess the data
    logging.info("Preprocessing data...")
    df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' to datetime
    df['Month'] = df['Date'].dt.month       # Extract month
    df['Year'] = df['Date'].dt.year         # Extract year

    # Define holiday flags
    logging.info("Defining seasonal flags...")
    df['Christmas'] = df['Month'].apply(lambda x: 1 if x == 12 else 0)  # Christmas in December
    df['Easter'] = df['Date'].apply(
        lambda x: 1 if x in pd.to_datetime(['2022-04-17', '2023-04-09', '2024-03-31']) else 0
    )  # Easter dates (adjust as needed)

    # Aggregate sales by month and holiday flags
    logging.info("Aggregating sales data by month and holiday flags...")
    monthly_sales = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()

    # Compare sales for Christmas and Easter
    christmas_sales = df[df['Christmas'] == 1]['Sales'].sum()
    easter_sales = df[df['Easter'] == 1]['Sales'].sum()
    total_sales = df['Sales'].sum()

    # Log seasonal sales
    logging.info(f"Christmas Sales: {christmas_sales}")
    logging.info(f"Easter Sales: {easter_sales}")
    logging.info(f"Total Sales: {total_sales}")

    # Visualize sales trends by month
    logging.info("Visualizing sales trends by month...")
    plt.figure(figsize=(12, 6))
    for year in monthly_sales['Year'].unique():
        yearly_sales = monthly_sales[monthly_sales['Year'] == year]
        plt.plot(yearly_sales['Month'], yearly_sales['Sales'], marker='o', label=f"Year {year}")

    plt.title("Monthly Sales Trends")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.xticks(range(1, 13), [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ])
    plt.legend()
    plt.grid()
    plt.savefig("monthly_sales_trends.png")
    plt.show()

    # Highlight Christmas and Easter Sales in the Visualization
    logging.info("Visualizing Christmas and Easter sales...")
    holiday_sales = df.groupby(['Month', 'Christmas', 'Easter'])['Sales'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    plt.bar(holiday_sales['Month'], holiday_sales['Sales'], color=[
        'red' if row['Christmas'] == 1 else 'blue' if row['Easter'] == 1 else 'gray'
        for _, row in holiday_sales.iterrows()
    ])
    plt.title("Holiday Sales Comparison")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.xticks(range(1, 13), [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ])
    plt.savefig("holiday_sales_comparison.png")
    plt.show()

except Exception as e:
    logging.error(f"An error occurred: {e}")
    print(f"An error occurred. Check the log file for details.")

import seaborn as sns

# Set up logging
logging.basicConfig(
    filename="correlation_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# File path for train.csv
data = r"C:\Users\Hasan\Desktop\week4"
train_file = os.path.join(data, "train.csv")

try:
    # Load the dataset
    logging.info("Loading train.csv...")
    df = pd.read_csv(train_file)
    logging.info(f"train.csv loaded successfully. Shape: {df.shape}")

    # Check for missing values in 'Sales' and 'Customers'
    logging.info("Checking for missing values in 'Sales' and 'Customers'...")
    missing_sales = df['Sales'].isnull().sum()
    missing_customers = df['Customers'].isnull().sum()
    logging.info(f"Missing Sales: {missing_sales}, Missing Customers: {missing_customers}")

    # Drop rows with missing values
    if missing_sales > 0 or missing_customers > 0:
        df = df.dropna(subset=['Sales', 'Customers'])
        logging.info("Dropped rows with missing 'Sales' or 'Customers'.")

    # Calculate correlation
    logging.info("Calculating correlation between 'Sales' and 'Customers'...")
    correlation = df['Sales'].corr(df['Customers'])
    logging.info(f"Correlation coefficient: {correlation:.2f}")
    print(f"Correlation coefficient between Sales and Customers: {correlation:.2f}")

    # Visualize the relationship
    logging.info("Visualizing the relationship between 'Sales' and 'Customers'...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Customers', y='Sales', alpha=0.6, color='blue')
    plt.title(f"Scatter Plot of Sales vs. Customers\nCorrelation Coefficient: {correlation:.2f}")
    plt.xlabel("Number of Customers")
    plt.ylabel("Sales")
    sns.regplot(data=df, x='Customers', y='Sales', scatter=False, color='red')
    plt.savefig("sales_vs_customers_correlation.png")
    plt.show()
    logging.info("Scatter plot saved as 'sales_vs_customers_correlation.png'.")

except Exception as e:
    logging.error(f"An error occurred: {e}")
    print(f"An error occurred. Check the log file for details.")
#To analyze how promotions (e.g., the Promo column) affect sales and whether they attract more customers or primarily increase sales from existing customers
# Set up logging
logging.basicConfig(
    filename="promo_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# File path for train.csv
data = r"C:\Users\Hasan\Desktop\week4"
train = os.path.join(data, "train.csv")

try:
    # Load the dataset
    logging.info("Loading train.csv...")
    df = pd.read_csv(train_file)
    logging.info(f"train.csv loaded successfully. Shape: {df.shape}")

    # Check for missing values in Promo, Sales, and Customers
    logging.info("Checking for missing values in 'Promo', 'Sales', and 'Customers'...")
    missing_values = df[['Promo', 'Sales', 'Customers']].isnull().sum()
    logging.info(f"Missing values:\n{missing_values}")
    df = df.dropna(subset=['Promo', 'Sales', 'Customers'])  # Drop rows with missing values

    # Group data by Promo
    logging.info("Grouping data by 'Promo'...")
    promo_analysis = df.groupby('Promo').agg(
        TotalSales=('Sales', 'sum'),
        AvgSales=('Sales', 'mean'),
        TotalCustomers=('Customers', 'sum'),
        AvgCustomers=('Customers', 'mean'),
        AvgSalesPerCustomer=('Sales', lambda x: (x / df.loc[x.index, 'Customers']).mean())
    ).reset_index()

    logging.info(f"Promo Analysis:\n{promo_analysis}")
    print("\nPromo Analysis:")
    print(promo_analysis)    

   
except Exception as e:
    logging.error(f"An error occurred: {e}")
    print(f"An error occurred. Check the log file for details.")

# Set up logging
logging.basicConfig(
    filename="promo_analysis_alt_graphs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# File path for train.csv
data = r"C:\Users\Hasan\Desktop\week4"
train_file = os.path.join(data, "train.csv")

try:
    # Load the dataset
    logging.info("Loading train.csv...")
    df = pd.read_csv(train_file)
    logging.info(f"train.csv loaded successfully. Shape: {df.shape}")

    # Drop rows with missing values in relevant columns
    logging.info("Dropping rows with missing values in 'Promo', 'Sales', and 'Customers'...")
    df = df.dropna(subset=['Promo', 'Sales', 'Customers'])

    # Box Plot: Sales and Customers Distribution by Promo
    logging.info("Creating box plots for Sales and Customers by Promo...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Promo', y='Sales', palette='Set2')
    plt.title("Box Plot of Sales by Promo")
    plt.xlabel("Promo (0 = No Promo, 1 = Promo)")
    plt.ylabel("Sales")
    plt.savefig("boxplot_sales_by_promo.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Promo', y='Customers', palette='Set3')
    plt.title("Box Plot of Customers by Promo")
    plt.xlabel("Promo (0 = No Promo, 1 = Promo)")
    plt.ylabel("Customers")
    plt.savefig("boxplot_customers_by_promo.png")
    plt.show()

    # Histogram: Frequency Distribution of Sales and Customers
    logging.info("Creating histograms for Sales and Customers by Promo...")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Sales', hue='Promo', kde=True, bins=30, palette='coolwarm', element='step')
    plt.title("Sales Distribution by Promo")
    plt.xlabel("Sales")
    plt.ylabel("Frequency")
    plt.savefig("histogram_sales_by_promo.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Customers', hue='Promo', kde=True, bins=30, palette='coolwarm', element='step')
    plt.title("Customers Distribution by Promo")
    plt.xlabel("Customers")
    plt.ylabel("Frequency")
    plt.savefig("histogram_customers_by_promo.png")
    plt.show()

    # Line Chart: Sales and Customers Trends Over Time
    logging.info("Creating line charts for Sales and Customers trends over time by Promo...")
    df['Date'] = pd.to_datetime(df['Date'])  # Convert Date to datetime
    daily_sales = df.groupby(['Date', 'Promo'])['Sales'].sum().reset_index()
    daily_customers = df.groupby(['Date', 'Promo'])['Customers'].sum().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_sales, x='Date', y='Sales', hue='Promo', palette='husl')
    plt.title("Daily Sales Trends by Promo")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.savefig("line_sales_trends_by_promo.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_customers, x='Date', y='Customers', hue='Promo', palette='husl')
    plt.title("Daily Customer Trends by Promo")
    plt.xlabel("Date")
    plt.ylabel("Total Customers")
    plt.savefig("line_customers_trends_by_promo.png")
    plt.show()

    
except Exception as e:
    logging.error(f"An error occurred: {e}")
    print(f"An error occurred. Check the log file for details.")
#To determine if promotions could be deployed more effectively and which stores should receive them
# Set up logging
logging.basicConfig(
    filename="promo_store_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# File path for train.csv
data = r"C:\Users\Hasan\Desktop\week4"
train_file = os.path.join(data, "train.csv")

try:
    # Load the dataset
    logging.info("Loading train.csv...")
    df = pd.read_csv(train_file)
    logging.info(f"train.csv loaded successfully. Shape: {df.shape}")

    # Drop rows with missing values in relevant columns
    logging.info("Dropping rows with missing values in 'Promo', 'Sales', and 'Customers'...")
    df = df.dropna(subset=['Promo', 'Sales', 'Customers'])

    # Group by Store and Promo to analyze performance
    logging.info("Grouping data by 'Store' and 'Promo' for analysis...")
    store_promo_analysis = df.groupby(['Store', 'Promo']).agg(
        TotalSales=('Sales', 'sum'),
        AvgSales=('Sales', 'mean'),
        TotalCustomers=('Customers', 'sum'),
        AvgCustomers=('Customers', 'mean')
    ).reset_index()

    # Calculate differences in sales and customers with and without promotions
    logging.info("Calculating differences in sales and customers with/without promotions...")
    promo_effectiveness = store_promo_analysis.pivot(index='Store', columns='Promo', values=['TotalSales', 'TotalCustomers']).reset_index()
    promo_effectiveness.columns = ['Store', 'NoPromo_Sales', 'Promo_Sales', 'NoPromo_Customers', 'Promo_Customers']
    promo_effectiveness['SalesIncrease'] = promo_effectiveness['Promo_Sales'] - promo_effectiveness['NoPromo_Sales']
    promo_effectiveness['CustomerIncrease'] = promo_effectiveness['Promo_Customers'] - promo_effectiveness['NoPromo_Customers']

    # Highlight stores with significant increases
    top_performers = promo_effectiveness[promo_effectiveness['SalesIncrease'] > 0].sort_values(by='SalesIncrease', ascending=False)
    underperformers = promo_effectiveness[promo_effectiveness['SalesIncrease'] <= 0]

    logging.info(f"Top-performing stores (Promo Effective):\n{top_performers.head()}")
    logging.info(f"Underperforming stores (Promo Ineffective):\n{underperformers.head()}")

    # Visualize effectiveness
    logging.info("Visualizing promo effectiveness by store...")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_performers, x='Store', y='SalesIncrease', color='green')
    plt.title("Sales Increase Due to Promotions (Top Stores)")
    plt.xlabel("Store")
    plt.ylabel("Sales Increase")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("promo_sales_increase_top_stores.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=underperformers, x='Store', y='SalesIncrease', color='red')
    plt.title("Sales Impact of Promotions (Underperforming Stores)")
    plt.xlabel("Store")
    plt.ylabel("Sales Increase")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("promo_sales_underperforming_stores.png")
    plt.show()

    # Recommendations based on analysis
    logging.info("Generating recommendations...")
    print("\nTop-performing stores for promotions (deploy here):")
    print(top_performers[['Store', 'SalesIncrease', 'CustomerIncrease']].head())

    print("\nUnderperforming stores (consider re-evaluating promo strategy):")
    print(underperformers[['Store', 'SalesIncrease', 'CustomerIncrease']].head())

except Exception as e:
    logging.error(f"An error occurred: {e}")
    print(f"An error occurred. Check the log file for details.")
#To analyze customer behavior trends during store opening and closing times
# Set up logging
logging.basicConfig(
    filename="customer_behavior_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# File path for train.csv
data = r"C:\Users\Hasan\Desktop\week4"
train_file = os.path.join(data, "train.csv")

try:
    # Load the dataset
    logging.info("Loading train.csv...")
    df = pd.read_csv(train_file)
    logging.info(f"train.csv loaded successfully. Shape: {df.shape}")

    # Drop rows with missing values in relevant columns
    logging.info("Dropping rows with missing values in 'Open' and 'Customers'...")
    df = df.dropna(subset=['Open', 'Customers'])

    # Analyze customer behavior when stores are open and closed
    logging.info("Analyzing customer behavior during store open and closed periods...")
    open_closed_analysis = df.groupby('Open')['Customers'].agg(
        TotalCustomers='sum',
        AvgCustomers='mean',
        Count='count'
    ).reset_index()

    logging.info(f"Customer behavior analysis:\n{open_closed_analysis}")
    print("\nCustomer Behavior Analysis:")
    print(open_closed_analysis)

    # Visualize customer trends during store open/closed times
    logging.info("Visualizing customer trends during store open and closed periods...")
    plt.figure(figsize=(8, 6))
    sns.barplot(data=open_closed_analysis, x='Open', y='TotalCustomers', palette='coolwarm')
    plt.title("Total Customers: Store Open vs Closed")
    plt.xlabel("Open (0 = Closed, 1 = Open)")
    plt.ylabel("Total Customers")
    plt.savefig("total_customers_open_closed.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.barplot(data=open_closed_analysis, x='Open', y='AvgCustomers', palette='coolwarm')
    plt.title("Average Customers: Store Open vs Closed")
    plt.xlabel("Open (0 = Closed, 1 = Open)")
    plt.ylabel("Average Customers")
    plt.savefig("avg_customers_open_closed.png")
    plt.show()

    # Analyze customer trends over time if timestamps are available
    if 'Date' in df.columns:
        logging.info("Analyzing customer trends over time...")
        df['Date'] = pd.to_datetime(df['Date'])
        daily_trends = df.groupby(['Date', 'Open'])['Customers'].sum().reset_index()

        # Line plot for customer trends over time
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=daily_trends, x='Date', y='Customers', hue='Open', palette='husl')
        plt.title("Daily Customer Trends: Store Open vs Closed")
        plt.xlabel("Date")
        plt.ylabel("Total Customers")
        plt.savefig("daily_customer_trends.png")
        plt.show()

except Exception as e:
    logging.error(f"An error occurred: {e}")
    print(f"An error occurred. Check the log file for details.")
#To determine which stores are open on all weekdays and how this affects their sales on weekends
# Set up logging
logging.basicConfig(
    filename="weekday_weekend_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# File path for train.csv
data = r"C:\Users\Hasan\Desktop\week4"
train_file = os.path.join(data, "train.csv")

try:
    # Load the dataset
    logging.info("Loading train.csv...")
    df = pd.read_csv(train_file)
    logging.info(f"train.csv loaded successfully. Shape: {df.shape}")

    # Drop rows with missing values in relevant columns
    logging.info("Dropping rows with missing values in 'Open' and 'Sales'...")
    df = df.dropna(subset=['Open', 'Sales', 'DayOfWeek'])

    # Identify stores open on all weekdays
    logging.info("Identifying stores open on all weekdays...")
    weekday_open = df[df['DayOfWeek'].isin([1, 2, 3, 4, 5]) & (df['Open'] == 1)]
    weekday_open_counts = weekday_open.groupby('Store')['DayOfWeek'].nunique()
    stores_open_all_weekdays = weekday_open_counts[weekday_open_counts == 5].index.tolist()
    logging.info(f"Stores open on all weekdays: {stores_open_all_weekdays}")

    # Flag stores as open_all_weekdays
    df['OpenAllWeekdays'] = df['Store'].apply(lambda x: 1 if x in stores_open_all_weekdays else 0)

    # Filter weekend data
    logging.info("Filtering weekend data...")
    weekend_data = df[df['DayOfWeek'].isin([6, 7])]

    # Analyze weekend sales for both groups
    logging.info("Analyzing weekend sales for stores open on all weekdays vs others...")
    weekend_analysis = weekend_data.groupby('OpenAllWeekdays')['Sales'].agg(
        TotalSales='sum',
        AvgSales='mean',
        StoreCount='count'
    ).reset_index()

    logging.info(f"Weekend sales analysis:\n{weekend_analysis}")
    print("\nWeekend Sales Analysis:")
    print(weekend_analysis)

    # Visualize weekend sales
    logging.info("Visualizing weekend sales comparison...")
    plt.figure(figsize=(8, 6))
    sns.barplot(data=weekend_analysis, x='OpenAllWeekdays', y='TotalSales', palette='coolwarm')
    plt.title("Total Weekend Sales: Open All Weekdays vs Others")
    plt.xlabel("Open All Weekdays (0 = No, 1 = Yes)")
    plt.ylabel("Total Weekend Sales")
    plt.xticks([0, 1], ["Not Open All Weekdays", "Open All Weekdays"])
    plt.savefig("total_weekend_sales_open_all_weekdays.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.barplot(data=weekend_analysis, x='OpenAllWeekdays', y='AvgSales', palette='coolwarm')
    plt.title("Average Weekend Sales: Open All Weekdays vs Others")
    plt.xlabel("Open All Weekdays (0 = No, 1 = Yes)")
    plt.ylabel("Average Weekend Sales")
    plt.xticks([0, 1], ["Not Open All Weekdays", "Open All Weekdays"])
    plt.savefig("avg_weekend_sales_open_all_weekdays.png")
    plt.show()

except Exception as e:
    logging.error(f"An error occurred: {e}")
    print(f"An error occurred. Check the log file for details.")
#To analyze how the assortment type affects sales
# Set up logging
logging.basicConfig(
    filename="assortment_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# File path for store.csv
data = r"C:\Users\Hasan\Desktop\week4"
store_file = os.path.join(data, "store.csv")
train_file = os.path.join(data, "train.csv")

try:
    # Load the store dataset
    logging.info("Loading store.csv...")
    store_df = pd.read_csv(store_file)
    logging.info(f"store.csv loaded successfully. Shape: {store_df.shape}")

    # Load the train dataset
    logging.info("Loading train.csv...")
    train_df = pd.read_csv(train_file)
    logging.info(f"train.csv loaded successfully. Shape: {train_df.shape}")

    # Merge train and store datasets on Store
    logging.info("Merging train.csv and store.csv on 'Store'...")
    merged_df = pd.merge(train_df, store_df, on='Store', how='left')

    # Drop rows with missing values in 'Assortment' and 'Sales'
    logging.info("Dropping rows with missing values in 'Assortment' and 'Sales'...")
    merged_df = merged_df.dropna(subset=['Assortment', 'Sales'])

    # Analyze sales by assortment type
    logging.info("Grouping data by 'Assortment' to analyze sales...")
    assortment_analysis = merged_df.groupby('Assortment')['Sales'].agg(
        TotalSales='sum',
        AvgSales='mean',
        StoreCount='count'
    ).reset_index()

    logging.info(f"Assortment sales analysis:\n{assortment_analysis}")
    print("\nAssortment Sales Analysis:")
    print(assortment_analysis)

    # Visualize total and average sales by assortment
    logging.info("Visualizing total and average sales by assortment type...")
    plt.figure(figsize=(8, 6))
    sns.barplot(data=assortment_analysis, x='Assortment', y='TotalSales', palette='Set2')
    plt.title("Total Sales by Assortment Type")
    plt.xlabel("Assortment Type")
    plt.ylabel("Total Sales")
    plt.savefig("total_sales_by_assortment.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.barplot(data=assortment_analysis, x='Assortment', y='AvgSales', palette='Set3')
    plt.title("Average Sales by Assortment Type")
    plt.xlabel("Assortment Type")
    plt.ylabel("Average Sales")
    plt.savefig("avg_sales_by_assortment.png")
    plt.show()

    # Optional: Analyze sales trends over time by assortment type
    if 'Date' in train_df.columns:
        logging.info("Analyzing sales trends over time by assortment type...")
        merged_df['Date'] = pd.to_datetime(merged_df['Date'])
        daily_sales = merged_df.groupby(['Date', 'Assortment'])['Sales'].sum().reset_index()

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=daily_sales, x='Date', y='Sales', hue='Assortment', palette='husl')
        plt.title("Sales Trends Over Time by Assortment Type")
        plt.xlabel("Date")
        plt.ylabel("Total Sales")
        plt.legend(title='Assortment')
        plt.savefig("sales_trends_by_assortment.png")
        plt.show()

except Exception as e:
    logging.error(f"An error occurred: {e}")
    print(f"An error occurred. Check the log file for details.")

# Set up logging
logging.basicConfig(
    filename="competition_distance_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# File paths
data = r"C:\Users\Hasan\Desktop\week4"
store_file = os.path.join(data, "store.csv")
train_file = os.path.join(data, "train.csv")

try:
    # Load datasets
    logging.info("Loading store.csv and train.csv...")
    store_df = pd.read_csv(store_file)
    train_df = pd.read_csv(train_file)

    # Merge datasets on 'Store'
    logging.info("Merging datasets on 'Store'...")
    merged_df = pd.merge(train_df, store_df, on='Store', how='left')

    # Drop rows with missing values in relevant columns
    logging.info("Dropping rows with missing values in 'CompetitionDistance' and 'Sales'...")
    merged_df = merged_df.dropna(subset=['CompetitionDistance', 'Sales'])

    # Analyze sales vs competition distance
    logging.info("Grouping data by 'CompetitionDistance' range to analyze sales...")
    merged_df['CompetitionDistanceRange'] = pd.cut(
        merged_df['CompetitionDistance'],
        bins=[0, 500, 1000, 5000, 10000, float('inf')],
        labels=['0-500', '500-1000', '1000-5000', '5000-10000', '10k+']
    )
    distance_analysis = merged_df.groupby('CompetitionDistanceRange')['Sales'].agg(
        TotalSales='sum',
        AvgSales='mean'
    ).reset_index()

    logging.info(f"Sales vs Competition Distance Analysis:\n{distance_analysis}")
    print("\nSales vs Competition Distance Analysis:")
    print(distance_analysis)

    # Visualize sales vs competition distance
    logging.info("Visualizing sales vs competition distance...")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=distance_analysis, x='CompetitionDistanceRange', y='AvgSales', palette='coolwarm')
    plt.title("Average Sales by Competition Distance Range")
    plt.xlabel("Competition Distance Range (meters)")
    plt.ylabel("Average Sales")
    plt.savefig("avg_sales_by_competition_distance.png")
    plt.show()

    # Analyze city-center stores (assuming CityCenter flag or similar exists)
    if 'StoreType' in store_df.columns:  # Adjust based on actual column
        logging.info("Analyzing competition distance effect for city-center stores...")
        merged_df['CityCenter'] = merged_df['StoreType'].apply(lambda x: 1 if x == 'C' else 0)
        
        city_center_analysis = merged_df[merged_df['CityCenter'] == 1].groupby('CompetitionDistanceRange')['Sales'].agg(
            TotalSales='sum',
            AvgSales='mean'
        ).reset_index()

        logging.info(f"City-Center Store Analysis:\n{city_center_analysis}")
        print("\nCity-Center Store Analysis:")
        print(city_center_analysis)

        # Visualize city-center sales vs competition distance
        logging.info("Visualizing city-center sales vs competition distance...")
        plt.figure(figsize=(10, 6))
        sns.barplot(data=city_center_analysis, x='CompetitionDistanceRange', y='AvgSales', palette='Set2')
        plt.title("City-Center Stores: Average Sales by Competition Distance Range")
        plt.xlabel("Competition Distance Range (meters)")
        plt.ylabel("Average Sales")
        plt.savefig("city_center_sales_by_competition_distance.png")
        plt.show()

except Exception as e:
    logging.error(f"An error occurred: {e}")
    print(f"An error occurred. Check the log file for details.")
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Set the path to your data directory
data = r"C:\Users\Hasan\Desktop\week4"

# Define file paths
store_file = os.path.join(data, "store.csv")
train_file = os.path.join(data, "train.csv")

# Load datasets
store_data = pd.read_csv(store_file)
train_data = pd.read_csv(train_file, parse_dates=['Date'])

# Merge train and store data
df = train_data.merge(store_data, how='left', on='Store')

# 1. Extract datetime features
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Example holidays (customize as needed)
holidays = pd.to_datetime(['2023-12-25', '2024-01-01'])

df['DaysToNextHoliday'] = df['Date'].apply(lambda x: min([(h - x).days for h in holidays if h >= x], default=0))
df['DaysSinceLastHoliday'] = df['Date'].apply(lambda x: min([(x - h).days for h in holidays if h <= x], default=0))

# 2. Encode categorical features
df = pd.get_dummies(df, columns=['StoreType', 'Assortment'], drop_first=True)

# Encode PromoInterval
df['PromoInterval'] = df['PromoInterval'].fillna('None')
promo_interval_mapping = {'None': 0, 'Jan,Apr,Jul,Oct': 1, 'Feb,May,Aug,Nov': 2, 'Mar,Jun,Sept,Dec': 3}
df['PromoInterval'] = df['PromoInterval'].map(promo_interval_mapping)

# 3. Handle missing values
df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0).astype(int)
df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0).astype(int)

# 4. Create lag and rolling features
df['SalesLag1'] = df['Sales'].shift(1)
df['SalesRollingMean'] = df['Sales'].rolling(window=7).mean()

# 5. Normalize CompetitionDistance
scaler = MinMaxScaler()
df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].max())
df['CompetitionDistanceScaled'] = scaler.fit_transform(df[['CompetitionDistance']])

# 6. Prepare features and target
X = df.drop(['Sales', 'Date'], axis=1)
y = df['Sales']

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display a sample of the preprocessed data
print(df.head())

# 8. Save the processed data
output_file = os.path.join(data, "preprocessed_rossmann_data.csv")
df.to_csv(output_file, index=False)
print(f"Preprocessed data saved to: {output_file}")
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Set the path to your data directory
data = r"C:\Users\Hasan\Desktop\week4"

# Define file paths
store_file = os.path.join(data, "store.csv")
train_file = os.path.join(data, "train.csv")

# Load datasets
store_data = pd.read_csv(store_file)
train_data = pd.read_csv(train_file, parse_dates=['Date'])

# Merge train and store data
df = train_data.merge(store_data, how='left', on='Store')

# 1. Extract datetime features
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Extract additional date-based features
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)

# Feature: Beginning, Mid, and End of Month
df['MonthPeriod'] = pd.cut(df['Day'], bins=[0, 10, 20, 31], labels=['Beginning', 'Mid', 'End'])

# Encode MonthPeriod as numerical for modeling
month_period_mapping = {'Beginning': 0, 'Mid': 1, 'End': 2}
df['MonthPeriod'] = df['MonthPeriod'].map(month_period_mapping)

# Example holidays (customize as needed)
holidays = pd.to_datetime(['2023-12-25', '2024-01-01'])

df['DaysToNextHoliday'] = df['Date'].apply(lambda x: min([(h - x).days for h in holidays if h >= x], default=0))
df['DaysSinceLastHoliday'] = df['Date'].apply(lambda x: min([(x - h).days for h in holidays if h <= x], default=0))

# 2. Encode categorical features
df = pd.get_dummies(df, columns=['StoreType', 'Assortment'], drop_first=True)
state_holiday_mapping = {'0': 0, 'a': 1, 'b': 2, 'c': 3}
df['StateHoliday'] = df['StateHoliday'].map(state_holiday_mapping).fillna(0).astype(int)

# Encode PromoInterval
df['PromoInterval'] = df['PromoInterval'].fillna('None')
promo_interval_mapping = {'None': 0, 'Jan,Apr,Jul,Oct': 1, 'Feb,May,Aug,Nov': 2, 'Mar,Jun,Sept,Dec': 3}
df['PromoInterval'] = df['PromoInterval'].map(promo_interval_mapping)

# 3. Handle missing values
df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0).astype(int)
df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0).astype(int)

# 4. Create lag and rolling features
df['SalesLag1'] = df['Sales'].shift(1)
df['SalesRollingMean'] = df['Sales'].rolling(window=7).mean()

# 5. Normalize CompetitionDistance
df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].max())

# 6. Prepare features and target
X = df.drop(['Sales', 'Date'], axis=1)
y = df['Sales']

# 7. Scale the data using StandardScaler
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Display a sample of the preprocessed data
print(X_scaled.head())

# 9. Save the processed data
output_file = os.path.join(data, "preprocessed_rossmann_data.csv")
X_scaled['Sales'] = y  # Add target back for saving the final dataset
X_scaled.to_csv(output_file, index=False)
print(f"Preprocessed data saved to: {output_file}")
# Extract date features
# Extract date features
# Extract date features
# Identify numeric and categorical columns
numeric_features = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 
                    'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear', 
                    'Year', 'Month', 'Day', 'Weekday']
categorical_features = ['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval']

# Preprocessing for numeric data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors in a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_estimators=100))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
for col in df.columns:
    if df[col].dtype == 'object':
        unique_types = set(type(val) for val in df[col].dropna())
        if len(unique_types) > 1:
            print(f"Column '{col}' has mixed types: {unique_types}")
for col in df.columns:
    if df[col].dtype == 'object':
        unique_types = set(type(val) for val in df[col].dropna())
        if len(unique_types) > 1:
            print(f"Column '{col}' has mixed types: {unique_types}")
for col in df.columns:
    if df[col].dtype == 'object':
        unique_types = set(type(val) for val in df[col].dropna())
        if len(unique_types) > 1:
            print(f"Column '{col}' has mixed types: {unique_types}")
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
from sklearn.metrics import mean_squared_error
import numpy as np

# Calculate MSE and then take the square root to get RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Calculate R-squared
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R2: {r2}")

# Check if there are datetime columns
datetime_columns = X_train.select_dtypes(include=['datetime64']).columns
print("Datetime columns:", datetime_columns)

# Convert datetime columns to numeric features
for col in datetime_columns:
    X_train[col + '_year'] = X_train[col].dt.year
    X_train[col + '_month'] = X_train[col].dt.month
    X_train[col + '_day'] = X_train[col].dt.day
    X_train[col + '_weekday'] = X_train[col].dt.weekday

# Drop original datetime columns
X_train = X_train.drop(columns=datetime_columns)
# Repeat transformation for X_test
for col in datetime_columns:
    X_test[col + '_year'] = X_test[col].dt.year
    X_test[col + '_month'] = X_test[col].dt.month
    X_test[col + '_day'] = X_test[col].dt.day
    X_test[col + '_weekday'] = X_test[col].dt.weekday

# Drop original datetime columns in X_test
X_test = X_test.drop(columns=datetime_columns)
from sklearn.ensemble import RandomForestRegressor

# Assuming X_train is your prepared training data and y_train is your target variable
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
import numpy as np
import matplotlib.pyplot as plt

# Extract feature importances
feature_importances = model.feature_importances_
feature_names = X_train.columns  # If X_train is a pandas DataFrame

# Sort the features by importance
sorted_idx = np.argsort(feature_importances)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_names[sorted_idx], feature_importances[sorted_idx], align='center')
plt.xlabel('Feature Importance')
plt.title('Feature Importance from Random Forest')
plt.show()
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
import numpy as np
from sklearn.model_selection import train_test_split

# Assume X_train and y_train are already prepared

# Sample a subset of data for quicker iteration
X_sample, _, y_sample, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=42)

# Define the model
model = RandomForestRegressor(random_state=42)

# Define a reduced hyperparameter grid
param_distributions = {
    'n_estimators': [50],               # Fixed smaller number of trees
    'max_depth': [10],                  # Fixed depth for faster training
    'min_samples_split': [2, 5],        # Limited options
    'min_samples_leaf': [1, 2],         
    'max_features': ['sqrt']            # Avoid 'auto' and limit options
}

# Use RandomizedSearchCV for quick hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=model, 
    param_distributions=param_distributions, 
    n_iter=5,                           # Fewer random combinations
    cv=2,                               # Reduced cross-validation folds
    scoring='neg_mean_squared_error',    # Evaluation metric
    verbose=1, 
    random_state=42, 
    n_jobs=2                             # Limit parallel jobs
)

# Fit the model on the subset of data
random_search.fit(X_sample, y_sample)

# Output the best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best score:", -random_search.best_score_)

# Save the best model
dump(random_search.best_estimator_, 'optimized_random_forest_model_quick.pkl')
print("Model saved as optimized_random_forest_model_quick.pkl")
#for Serialization with Timestamp
from datetime import datetime
# Generate a timestamp for filename
timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S-%f')[:-3]
filename = f"optimized_random_forest_model_{timestamp}.pkl"

# Save the best model with a timestamp
dump(random_search.best_estimator_, filename)
print(f"Model saved as {filename}")
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import logging
from datetime import datetime
import os
# Set up logging
logging.basicConfig(
    filename=f'rossmann_model_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Started Rossmann LSTM Sales Prediction Process")

# Define dataset directory and file paths
data = r"C:\Users\Hasan\Desktop\week4"
train_file = os.path.join(data, "train.csv")
store_file = os.path.join(data, "store.csv")

logging.info(f"Data directory: {data}")
logging.info(f"Train file: {train_file}")
logging.info(f"Store file: {store_file}")
import pandas as pd
import logging
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Initialize logging
logging.basicConfig(level=logging.INFO)

try:
    # Load train data
    train_data = pd.read_csv(train_file, parse_dates=['Date'])
    
    # Convert 'StateHoliday' to numeric, assuming 0 for non-holiday and 1 for holiday
    train_data['StateHoliday'] = train_data['StateHoliday'].apply(lambda x: 1 if x != '0' else 0)
    
    # Group by 'Date' and sum the numeric columns, excluding 'StateHoliday'
    train_data = train_data.groupby('Date').sum().reset_index()

    logging.info("Train data loaded and processed successfully.")
except Exception as e:
    logging.error(f"Failed to load train data: {e}")
    raise

# Plot sales
plt.figure(figsize=(12, 6))
plt.plot(train_data['Date'], train_data['Sales'])
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Check stationarity
result = adfuller(train_data['Sales'])
logging.info(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
if result[1] < 0.05:
    logging.info("Data is stationary.")
else:
    logging.info("Data is not stationary. Applying differencing.")
    train_data['Sales'] = train_data['Sales'].diff().fillna(0)
import numpy as np
def create_supervised_data(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

window_size = 30
X, y = create_supervised_data(train_data['Sales'].values, window_size)
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X).reshape((X.shape[0], X.shape[1], 1))
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

logging.info("Data transformed into supervised format.")

#Build and Train LSTM Model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(window_size, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
logging.info("LSTM model compiled.")

# Train the model
try:
    model.fit(X_scaled, y_scaled, epochs=20, batch_size=32)
    logging.info("Model training completed.")
except Exception as e:
    logging.error(f"Model training failed: {e}")
    raise

         