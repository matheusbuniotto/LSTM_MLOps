import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow.pytorch
from mlflow import MlflowClient

client = MlflowClient()
mlflow.set_tracking_uri(uri="http://0.0.0.0:8081")
model_name = "lstm-2000-epochs"
model_version_alias = "champion"
client.set_registered_model_alias(model_name, model_version_alias, "1")

# Get information about the model
model_info = client.get_model_version_by_alias(model_name, model_version_alias)
model_tags = model_info.tags
print(model_tags)

model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlflow.pytorch.load_model(model_uri)

print(model)
