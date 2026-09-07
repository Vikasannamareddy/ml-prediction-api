import os
import joblib
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "churn_model.joblib"
)
def test_model_file_exists():
    assert os.path.exists(MODEL_PATH)
def test_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)
    assert model is not None