-- Databricks notebook source
-- MAGIC %md 
-- MAGIC ## Persist DLT streaming view
-- MAGIC To easily support DLT / UC / ML during the preview, we temporary recopy the final DLT view to another UC table 

-- COMMAND ----------

CREATE OR REPLACE TABLE yash_gupta_fsi_credit.yash_gupta_fsi_credit.customer_gold_features AS SELECT * FROM yash_gupta_fsi_credit.yash_gupta_fsi_credit.customer_gold;
CREATE OR REPLACE TABLE yash_gupta_fsi_credit.yash_gupta_fsi_credit.telco_gold_features AS SELECT * FROM yash_gupta_fsi_credit.yash_gupta_fsi_credit.telco_gold;
CREATE OR REPLACE TABLE yash_gupta_fsi_credit.yash_gupta_fsi_credit.fund_trans_gold_features AS SELECT * FROM yash_gupta_fsi_credit.yash_gupta_fsi_credit.fund_trans_gold;
CREATE OR REPLACE TABLE yash_gupta_fsi_credit.yash_gupta_fsi_credit.credit_bureau_gold_features AS SELECT * FROM yash_gupta_fsi_credit.yash_gupta_fsi_credit.credit_bureau_gold;
CREATE OR REPLACE TABLE yash_gupta_fsi_credit.yash_gupta_fsi_credit.customer_silver_features AS SELECT * FROM yash_gupta_fsi_credit.yash_gupta_fsi_credit.customer_silver;
