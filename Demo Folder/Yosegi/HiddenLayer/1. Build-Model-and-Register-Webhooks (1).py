# Databricks notebook source
# MAGIC %md <i18n value="8a133a94-0841-46ab-ad14-c85a27948a3c"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # MLflow Webhooks & Testing
# MAGIC
# MAGIC Webhooks trigger the execution of code (oftentimes tests) upon some event. 

# COMMAND ----------

# TODO
#Paste your token below
 
token = dbutils.secrets.get(scope="webhook-demo", key="dbx_access_token")

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pyspark.sql.functions as F
from delta.tables import *
import numpy as np

# With the token, we can create our authorization header for our subsequent REST calls
headers = {"Authorization": f"Bearer {token}"}

instance = mlflow.utils.databricks_utils.get_webapp_url()

# COMMAND ----------

# MAGIC %md <i18n value="086b4385-9eae-492e-8ccd-52d68a97ad86"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Train and Register a Model
# MAGIC
# MAGIC Build and log your model.

# COMMAND ----------

## Reading in data 
path = '/databricks-datasets/wine-quality/winequality-white.csv'
wine_df = (spark.read
           .option('header', 'true')
           .option('inferSchema', 'true')
           .option('sep', ';')
           .csv(path))
wine_df_clean = wine_df.select([F.col(col).alias(col.replace(' ', '_')) for col in wine_df.columns])
display(wine_df_clean)

# COMMAND ----------

# MAGIC %fs mkdirs /tmp/reproducible_ml_blog

# COMMAND ----------

## Write it out as a delta table 
write_path = 'dbfs:/tmp/reproducible_ml_blog/wine_quality_white.delta'
wine_df_clean.write.format('delta').mode('overwrite').option('overwriteSchema', 'true').save(write_path)

# COMMAND ----------

## Insert a new row 
new_row = spark.createDataFrame([[7, 0.27, 0.36, 1.6, 0.045, 45, 170, 1.001, 3, 0.45, 8.8, 6]])
wine_df_extra_row = wine_df_clean.union(new_row)
display(wine_df_extra_row)

# COMMAND ----------

## Write it out to delta location 
wine_df_extra_row.write.format('delta').mode('overwrite').option('overwriteSchema', 'true').save(write_path)

# COMMAND ----------

## Specifying data version to use for model training
version = 1 
wine_df_delta = spark.read.format('delta').option('versionAsOf', version).load(write_path).toPandas()
display(wine_df_delta)

# COMMAND ----------

## Split the data into training and test sets. (0.75, 0.25) split.
seed = 1111
train, test = train_test_split(wine_df_delta, train_size=0.75, random_state=seed)

## The target column is "quality" which is a scalar from [3, 9]
X_train = train.drop(['quality'], axis=1)
X_test = test.drop(['quality'], axis=1)
y_train = train[['quality']]
y_test = test[['quality']]

# COMMAND ----------

with mlflow.start_run() as run:
  ## Log your params
  n_estimators = 1000
  max_features = 'sqrt'
  params = {'data_version': version,
           'n_estimators': n_estimators,
           'max_features': max_features}
  mlflow.log_params(params)
  ## Train the model 
  rf = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=seed)
  rf.fit(X_train, y_train)

  ## Predict on the test data 
  preds = rf.predict(X_test)

  ## Generate metrics 
  rmse = np.sqrt(mean_squared_error(y_test, preds))
  mae = mean_absolute_error(y_test, preds)
  r2 = r2_score(y_test, preds)
  metrics = {'rmse': rmse,
             'mae': mae,
             'r2' : r2}

  ## Log Metrics
  mlflow.log_metrics(metrics)

  ## Log the model 
  mlflow.sklearn.log_model(rf, 'random-forest-model')  
  run_id = run.info.run_id
  experiment_id = run.info.experiment_id

# COMMAND ----------

# MAGIC %md <i18n value="8f56343e-2a5f-4515-be64-047b07dcf877"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Register the model

# COMMAND ----------

name = f"webhook-demo_wine_quality"
model_uri = f"runs:/{run_id}/random-forest-model"

model_details = mlflow.register_model(model_uri=model_uri, name=name)

# COMMAND ----------

# MAGIC %md <i18n value="02c615b7-dbf6-4e4a-8706-6c31cac2be68"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Creating the Job
# MAGIC
# MAGIC The following steps will create a Databricks job using another notebook in this directory: **`Webhooks-Job-Scan-Model`**

# COMMAND ----------

# MAGIC %md <i18n value="b22313af-97a9-43d8-aaf6-57755b3d45da"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Create a job that executes the notebook **`Webhooks-Job-Scan-Model`** in the same folder as this notebook.<br><br>
# MAGIC
# MAGIC - Hover over the sidebar in the Databricks UI on the left.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/ml-deployment/ClickWorkflows.png" alt="step12" width="150"/>
# MAGIC <br></br>
# MAGIC
# MAGIC - Click on Create Job
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/ml-deployment/CreateJob.png" alt="step12" width="750"/>
# MAGIC
# MAGIC <br></br>
# MAGIC - Name your Job
# MAGIC - Select the notebook **`Webhooks-Job-Scan-Model`** 
# MAGIC - Select the current cluster
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/ml-deployment/JobInfo.png" alt="step12" width="750"/>
# MAGIC
# MAGIC <br></br>
# MAGIC - Copy the Job ID
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/ml-deployment/JobID.png" alt="step12" width="450"/>

# COMMAND ----------

# MAGIC %md <i18n value="8c0aa70d-ab84-4e31-9ee4-2ab5d9fa6beb"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Create a Job Webhook
# MAGIC
# MAGIC There are a few different events that can trigger a Webhook. In this notebook, we will be experimenting with triggering a job when our model transitions between stages.

# COMMAND ----------

# MANUAL STEP: Set Job ID
job_id=503119664841490
# For demo purpose, keep the model scoped to the one we are building
name=f"webhook-demo_wine_quality"
token = dbutils.secrets.get(scope="webhook-demo", key="dbx_access_token")

# COMMAND ----------

# Create the webhook
import json
from mlflow.utils.rest_utils import http_request
from mlflow.utils.databricks_utils import get_databricks_host_creds

endpoint = "/api/2.0/mlflow/registry-webhooks/create"
host_creds = get_databricks_host_creds("databricks")

job_json = {"model_name": name,
            "events": ["MODEL_VERSION_TRANSITIONED_STAGE"],
            "description": "Job webhook trigger",
            "status": "Active",
            "job_spec": {"job_id": job_id,
                         "workspace_url": instance,
                         "access_token": token}
           }

response = http_request(
    host_creds=host_creds, 
    endpoint=endpoint,
    method="POST",
    json=job_json
)
assert response.status_code == 200, f"Expected HTTP 200, received {response.status_code}"

# COMMAND ----------

# MAGIC %md <i18n value="965cfc78-c346-40d2-a328-d3d769a8c3e2"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now that we have registered the webhook, we can **test it by transitioning our model from stage `None` to `Staging` in the Experiment UI.** We should see in the Jobs tab that our Job has run.

# COMMAND ----------

# List all webhooks
def list_webhooks_by_model_name(model_name):
  endpoint = f"/api/2.0/mlflow/registry-webhooks/list/?model_name={model_name}"
  response = http_request(
      host_creds=host_creds, 
      endpoint=endpoint,
      method="GET"
  )
  assert response.status_code == 200, f"Expected HTTP 200, received {response.status_code}"
  return response.json().get('webhooks', [])

hooks = list_webhooks_by_model_name(name)
print(hooks)

# COMMAND ----------

# Delete Webhooks, careful, this action is destructive
def delete_webhook_by_id(webhook_id):
    endpoint = f"/api/2.0/mlflow/registry-webhooks/delete"
    delete_json = {"id": webhook_id}

    print(f"Deleting Hook: {webhook_id}")
    response = http_request(
        host_creds=host_creds, 
        endpoint=endpoint,
        method="DELETE",
        json=delete_json
    )
    assert response.status_code == 200, f"Expected HTTP 200, received {response.status_code}"
    return response

for hook in list_webhooks_by_model_name(f"{name.replace(' ', '%20')}"):
  delete_webhook_by_id(hook.get('id'))


