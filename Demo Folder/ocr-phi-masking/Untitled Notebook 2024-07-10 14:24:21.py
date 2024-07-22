# Databricks notebook source
# List files in the specified directory
dbutils.fs.ls("dbfs:/home/yash.gupta@databricks.com/hls/phi_ocr/data/")

# COMMAND ----------

# MAGIC %%bash databricks fs cp dbfs:/home/yash.gupta@databricks.com/hls/phi_ocr/data/MT_OCR_00.pdf /local/path/MT_OCR_00.pdf
