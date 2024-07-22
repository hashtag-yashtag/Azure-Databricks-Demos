# Databricks notebook source
# MAGIC %run ./config

# COMMAND ----------

spark.sql("""CREATE CATALOG IF NOT EXISTS {catalog}.{schema}""")
spark.sql("""CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}""")

# COMMAND ----------

# Create the gold schema
spark.sql("""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.gold_financials (
    Ticker STRING,
    Security_Name STRING,
    Sector STRING,
    Industry STRING,
    Price DOUBLE,
    Market_Capitalization DOUBLE,
    Price_Earnings_Ratio DOUBLE,
    Forward_Price_Earnings DOUBLE,
    Price_Earnings_to_Growth DOUBLE,
    Price_to_Sales DOUBLE,
    Price_to_Book DOUBLE,
    Enterprise_Value_to_Revenue DOUBLE,
    Enterprise_Value_to_EBITDA DOUBLE,
    Profit_Margin DOUBLE,
    Operating_Margin DOUBLE,
    Return_on_Assets DOUBLE,
    Return_on_Equity DOUBLE,
    Revenue DOUBLE,
    Revenue_Per_Share DOUBLE,
    Quarterly_Revenue_Growth DOUBLE,
    Gross_Profit DOUBLE,
    EBITDA DOUBLE,
    Net_Income_Available_to_Common DOUBLE,
    Diluted_Earnings_Per_Share DOUBLE,
    Quarterly_Earnings_Growth DOUBLE,
    Total_Cash DOUBLE,
    Total_Cash_Per_Share DOUBLE,
    Total_Debt DOUBLE,
    Total_Debt_to_Equity DOUBLE,
    Current_Ratio DOUBLE,
    Book_Value_Per_Share DOUBLE,
    Operating_Cash_Flow DOUBLE,
    Levered_Free_Cash_Flow DOUBLE,
    Beta DOUBLE,
    Fifty_Two_Week_High DOUBLE,
    Fifty_Two_Week_Low DOUBLE,
    Fifty_Day_Moving_Average DOUBLE,
    Two_Hundred_Day_Moving_Average DOUBLE,
    Average_Volume DOUBLE,
    Shares_Outstanding DOUBLE,
    Float DOUBLE,
    Percentage_Held_by_Insiders DOUBLE,
    Percentage_Held_by_Institutions DOUBLE,
    Shares_Short DOUBLE,
    Short_Ratio DOUBLE,
    Short_Percentage_of_Float DOUBLE,
    Shares_Short_Prior_Month DOUBLE,
    Forward_Annual_Dividend_Rate DOUBLE,
    Forward_Annual_Dividend_Yield DOUBLE,
    Trailing_Annual_Dividend_Rate DOUBLE,
    Trailing_Annual_Dividend_Yield DOUBLE
)
""")
