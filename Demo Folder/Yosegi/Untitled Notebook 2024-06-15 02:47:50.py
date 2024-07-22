# Databricks notebook source
# MAGIC %pip install dbdemos

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import dbdemos
dbdemos.install('llm-rag-chatbot', catalog='yosegi_demo', schema='rag_chatbot')

# COMMAND ----------

import dbdemos
dbdemos.install('llm-fine-tuning', catalog='yosegi_demo', schema='fine_tuning_demo')
