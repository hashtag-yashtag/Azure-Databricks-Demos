# Databricks notebook source
# MAGIC %md
# MAGIC Demo configuration. 
# MAGIC
# MAGIC If you wish to install the demo using another schema and database, it's best to do it with dbdemos:
# MAGIC
# MAGIC `dbdemos.install('xxx', catalog='xx', schema='xx')`

# COMMAND ----------

catalog = "yosegi_demo"
dbName = db = "fine_tuning_demo"
volume_name = "raw_training_data"
