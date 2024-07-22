# Databricks notebook source
# MAGIC %pip install sentence-transformers
# MAGIC %pip install scikit-learn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

gold_financials_df = spark.table("yash_gupta_hackerspace.automap_dlt.gold_financials")
gold_columns = gold_financials_df.columns

embeddings = model.encode(gold_financials_df.columns)

print(embeddings.shape)


# COMMAND ----------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load the tables into DataFrames
gold_financials_df = spark.table("yash_gupta_hackerspace.automap_demo.gold_financials")
bronze_sp500_financials_df = spark.table("yash_gupta_hackerspace.automap_demo.bronze_sp500_financials")

# Extract column names
gold_columns = gold_financials_df.columns
bronze_columns = bronze_sp500_financials_df.columns

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for column names
gold_embeddings = model.encode(gold_columns)
bronze_embeddings = model.encode(bronze_columns)

# Calculate cosine similarity
similarity_matrix = cosine_similarity(gold_embeddings, bronze_embeddings)

# Create a DataFrame for similarity results
similarity_df = pd.DataFrame(similarity_matrix, index=gold_columns, columns=bronze_columns)

# Find the most similar column for each gold column
most_similar = similarity_df.idxmax(axis=1)
similarity_scores = similarity_df.max(axis=1)

# Create a result DataFrame
result_df = pd.DataFrame({
    'gold_column': gold_columns,
    'bronze_column': most_similar,
    'similarity': similarity_scores
})

# Convert to Spark DataFrame
result_spark_df = spark.createDataFrame(result_df)

# Display the result
display(result_spark_df)

# COMMAND ----------

from pyspark.sql import Window
import pyspark.sql.functions as F

# Define a window specification to partition by bronze_column and order by similarity descending
window_spec = Window.partitionBy("bronze_column").orderBy(F.desc("similarity"))

# Add a row number to each row within the partition
result_spark_df = result_spark_df.withColumn("row_number", F.row_number().over(window_spec))

# Set similarity to null for rows where row_number is not 1
result_spark_df = result_spark_df.withColumn("similarity", 
                                             F.when(F.col("row_number") != 1, F.lit(None)).otherwise(F.col("similarity")))

# Drop the row_number column as it is no longer needed
result_spark_df = result_spark_df.drop("row_number")

# Display the result
display(result_spark_df)

# COMMAND ----------

# Save the result_spark_df into a silver table
result_spark_df.write.format("delta").mode("overwrite").saveAsTable("yash_gupta_hackerspace.automap_demo.financials_mapping")

# COMMAND ----------

from pyspark.sql.functions import col

# Read the bronze and mapping tables
bronze_df = spark.table("yash_gupta_hackerspace.automap_demo.bronze_sp500_financials")
mapping_df = spark.table("yash_gupta_hackerspace.automap_demo.financials_mapping").filter(col("similarity").isNotNull())

# Create a dictionary of column mappings
column_mappings = dict(mapping_df.select("bronze_column", "gold_column").collect())

# Select and rename columns from bronze_df based on the mapping
selected_columns = [col(bronze_col).alias(gold_col) for bronze_col, gold_col in column_mappings.items()]
gold_df = bronze_df.select(*selected_columns)

# Write the result to the gold table
#gold_df.write.mode("overwrite").saveAsTable("financials_gold")

display(gold_df)

# COMMAND ----------

# MAGIC %sql select * from yash_gupta_hackerspace.automap_demo.gold_financials
