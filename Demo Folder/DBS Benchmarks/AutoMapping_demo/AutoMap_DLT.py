# Databricks notebook source
# MAGIC %pip install sentence-transformers
# MAGIC %pip install scikit-learn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./Init

# COMMAND ----------

# MAGIC %md 
# MAGIC # AI based columnar mapping solution
# MAGIC <img src="https://github.com/hashtag-yashtag/Azure-Databricks-Demos/blob/main/AI%20Mapping%20Architecture.png?raw=true" style="width: 800px; margin-left: 10px">
# MAGIC

# COMMAND ----------

# MAGIC %md ### Load all libraries
# MAGIC At the core of this, we will be using DLT ***(Delta Live Tables)***.
# MAGIC
# MAGIC Within DLT, we will be using libraries such as Sentence Transformers, Pandas, pyspark

# COMMAND ----------

import dlt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import col
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load SP500 data from Volumes into Spark Dataframe
# MAGIC Spark dataframe can infer headers, schemas, etc.
# MAGIC We will also need to remove some invalid characters before moving forward.
# MAGIC
# MAGIC This becomes our Bronze layer

# COMMAND ----------

@dlt.table(comment="SP500 Bronze")
def sp500_bronze():
    file_path = "dbfs:/Volumes/yash_gupta_hackerspace/automap_demo/files-for-ingest/sp500-financials.csv"

    # Read the CSV file into a Spark DataFrame
    df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(file_path)

    # Rename columns to remove invalid characters
    for col_name in df.columns:
        new_col_name = col_name.replace(" ", "_").replace(",", "_").replace(";", "_").replace("{", "_")\
                            .replace("}", "_").replace("(", "_").replace(")", "_").replace("\n", "_")\
                            .replace("\t", "_").replace("=", "_")
        df = df.withColumnRenamed(col_name, new_col_name)

    # Create the bronze table
    return df

# COMMAND ----------

# MAGIC %md ### Retreiving the Target Schema
# MAGIC In this case, we use the schema from a gold table with various common columns

# COMMAND ----------

# retreive the Gold Table. We will use this as a reference for the mapping
@dlt.table(comment="Financial Data Model Schema")
def target_schema():
    target_schema_df = spark.table("yash_gupta_hackerspace.automap_demo.gold_financials")
    return target_schema_df

# COMMAND ----------

# MAGIC %md ### Sentance Transformers and Encoding
# MAGIC
# MAGIC We will be using ***all-MiniLM-L6-v2*** <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width=40px>
# MAGIC
# MAGIC This is a miniaturized language encoding model which maps sentences and paragraphs to a 384 dimensional dence vector space and can be used for clustering or semantic search.
# MAGIC
# MAGIC Read more: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# MAGIC
# MAGIC Using this transformer, we will calculate similarity between target and bronze

# COMMAND ----------

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

@dlt.table(comment="Similarity Matrix")
def similarity_matrix():

    target_schema_df = dlt.read("target_schema")
    sp500_bronze_df = dlt.read("sp500_bronze")

    target_schema_columns = target_schema_df.columns
    bronze_columns = sp500_bronze_df.columns

    target_schema_embeddings = model.encode(target_schema_columns)
    bronze_embeddings = model.encode(bronze_columns)

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(target_schema_embeddings, bronze_embeddings)
    
    # Create a DataFrame for similarity results
    target_schema_columns = dlt.read("target_schema").columns
    bronze_columns = dlt.read("sp500_bronze").columns
    similarity_df = pd.DataFrame(similarity_matrix, index=target_schema_columns, columns=bronze_columns)
    
    # Find the most similar column for each target schema column
    most_similar = similarity_df.idxmax(axis=1)
    similarity_scores = similarity_df.max(axis=1)
    
    # Create a result DataFrame
    result_df = pd.DataFrame({
        'target_schema_column': target_schema_columns,
        'bronze_column': most_similar,
        'similarity': similarity_scores
    })
    
    # Convert to Spark DataFrame
    result_spark_df = spark.createDataFrame(result_df)
    return result_spark_df

# COMMAND ----------

# MAGIC %md ### Mapping
# MAGIC Based on the Similarity Matrix, we pick the "Best-match" and create a **Mapping Table**
# MAGIC
# MAGIC This Mapping table will have
# MAGIC - Target Column Name
# MAGIC - Bronze Column Name
# MAGIC - Similarity
# MAGIC
# MAGIC It is advised that upon generation of this mapping table, a **Human be present in the loop** to ensure mapping is correct

# COMMAND ----------

@dlt.table(comment="sp500_mapping")
def mapping():
    # Define a window specification to partition by bronze_column and order by similarity descending
    window_spec = Window.partitionBy("bronze_column").orderBy(F.desc("similarity"))

    # Add a row number to each row within the partition
    similarity_matrix_df = dlt.read("similarity_matrix")
    result_spark_df = similarity_matrix_df.withColumn("row_number", F.row_number().over(window_spec))

    # Set similarity to null for rows where row_number is not 1
    result_spark_df = result_spark_df.withColumn("similarity", 
                                                F.when(F.col("row_number") != 1, F.lit(None)).otherwise(F.col("similarity")))

    # Drop the row_number column as it is no longer needed
    result_spark_df = result_spark_df.drop("row_number")
    return result_spark_df


# COMMAND ----------

# MAGIC %md ### Silver Table
# MAGIC Finally, we can use this mapping table to move data from our Bronze Table into Silver table.
# MAGIC
# MAGIC It is also advised that any additional complex mappings / ETLs be performed during this step

# COMMAND ----------

@dlt.table(comment="silver_financials")
def silver_sp500():
    bronze_df = dlt.read("sp500_bronze")
    mapping_df = dlt.read("mapping").filter(col("similarity").isNotNull())

    column_mappings = dict(mapping_df.select("bronze_column", "target_schema_column").collect())

    # Select and rename columns from bronze_df based on the mapping against target
    selected_columns = [col(bronze_col).alias(target_col) for bronze_col, target_col in column_mappings.items()]
    silver_df = bronze_df.select(*selected_columns)

    # Write the result to the silver table
    return silver_df


# COMMAND ----------

# MAGIC %md ### Not Included in Demo:
# MAGIC - Gold Table
# MAGIC - Entity Resolution
# MAGIC - More than one source
# MAGIC - Filtering
# MAGIC
# MAGIC These features are expected to be built specific to the use case.

# COMMAND ----------

# MAGIC %md #
