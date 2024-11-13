import os
import joblib

model_name = "random_forest_weed.joblib"
model_path = os.path.join("./models", model_name)

loaded_rf_model = joblib.load(model_path)

y_pred = loaded_rf_model.predict()
