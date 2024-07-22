# Databricks notebook source
# MAGIC %md <i18n value="da6eb3d9-8d66-4bd3-aa77-0eb4bcc5e5e5"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Load the model name. The **`event_message`** is automatically populated by the webhook.

# COMMAND ----------

# If this was in the DBR it would be much nicer.  Though we could always do this with vanilla python.
%pip install --upgrade git+https://github.com/hiddenlayerai/hiddenlayer-sdk-python
dbutils.library.restartPython()

# COMMAND ----------

from typing import Optional
import mlflow
import requests
import time
token = dbutils.secrets.get(scope="webhook-demo", key="dbx_access_token")
instance = mlflow.utils.databricks_utils.get_webapp_url()

# COMMAND ----------

import json
 
event_message = dbutils.widgets.get("event_message")
print(event_message)
event_message_dict = json.loads(event_message)
model_name = event_message_dict.get("model_name")

print(event_message_dict)
print(model_name)



# COMMAND ----------

model_name= "webhook-demo_wine_quality"

# COMMAND ----------

# MAGIC %md <i18n value="3ff7d618-4a2c-46b4-88f4-4a145075f6eb"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Use the model name to get the latest model version.

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
rm = client.get_registered_model(model_name)
version = rm.latest_versions[0].version
# Get search results filtered by the registered model name
filter_string = f"name='{model_name}'"
results = client.search_registered_models(filter_string=filter_string)
for res in results:
    for mv in res.latest_versions:
        print(f"name={mv.name}; run_id={mv.run_id}; version={mv.version}")
        run_id = mv.run_id
# grab artifact list
artifacts = client.list_artifacts(run_id)

# COMMAND ----------

# Download artifacts
import os
local_dir = f"/tmp/{model_name}-{version}-artifacts"

# Create the local directory if it doesn't exist
if not os.path.exists(local_dir):
  os.mkdir(local_dir)

# Download artifacts
for artifact in artifacts:
  local_path = client.download_artifacts(run_id, artifact.path, local_dir)
  #print(f"Artifacts: {os.listdir(local_path)}")
  #print(f"Artifacts downloaded in: {local_path}")



# COMMAND ----------

# MAGIC %md <i18n value="225097e4-01a8-4f30-a08a-14949d7ef152"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Here we Scan the model that is registered with model registry.

# COMMAND ----------

# Create HiddenLayer SDK Client
from hiddenlayer import HiddenlayerServiceClient

hl_client_id=dbutils.secrets.get(scope="webhook-demo", key="hl_client_id")
hl_secret=dbutils.secrets.get(scope="webhook-demo", key="hl_secret")

hl_client = HiddenlayerServiceClient(
  host="https://api.hiddenlayer.ai",
  api_id=f"{hl_client_id}", 
  api_key=f"{hl_secret}"
)


# COMMAND ----------

# Utility function to add a comment to a model
import json
from mlflow.utils.rest_utils import http_request
from mlflow.utils.databricks_utils import get_databricks_host_creds
host_creds = get_databricks_host_creds("databricks")

def set_tag_on_model(model_name, version, tag, value):
    endpoint = f"/api/2.0/mlflow/model-versions/set-tag"
    tag_json = {
      "name": model_name,
      "version": version,
      "key": tag,
      "value": value
    }

    response = http_request(
        host_creds=host_creds, 
        endpoint=endpoint,
        method="POST",
        json=tag_json
    )
    assert response.status_code == 200, f"Expected HTTP 200, received {response.status_code}"
    return response


def add_comment_to_model(model_name, version, comment):
    endpoint = f"/api/2.0/mlflow/comments/create"
    comment_json = {
      "name": model_name,
      "version": version,
      "comment": comment
    }

    response = http_request(
        host_creds=host_creds, 
        endpoint=endpoint,
        method="POST",
        json=comment_json
    )
    assert response.status_code == 200, f"Expected HTTP 200, received {response.status_code}"
    return response
  

# COMMAND ----------

import json
for file in os.listdir(local_path):
  # For this demo, only scan the pkl file
  if file.endswith(".pkl"): 
    print(f"Scanning: {local_path}/{file}")
    hl_client.model_scanner.scan_file(
      model_path=f"{local_path}/{file}", model_name=f"dbx-{model_name}", wait_for_results=True
    )  

# COMMAND ----------

from datetime import datetime
scan_results=hl_client.model_scanner.get_scan_results(model_name=f"dbx-{model_name}")

if len(scan_results.detections) == 0:
    conviction = "SAFE"
else:
    conviction = "UNSAFE"

now = datetime.now()
add_comment_to_model(model_name, version, f"HiddenLayer Scan Complete: [{conviction}]")
set_tag_on_model(model_name, version, "hl-scan-results", conviction)
set_tag_on_model(model_name, version, "hl-scan-updated-at", now.strftime("%m/%d/%Y %H:%M:%S"))
