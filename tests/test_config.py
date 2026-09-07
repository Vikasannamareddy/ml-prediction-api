import os
import json
THRESHOLD_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "threshold.json"
)
def test_threshold_file_exists():
    assert os.path.exists(THRESHOLD_PATH)
def test_threshold_is_valid():
    with open(THRESHOLD_PATH, "r") as f:
        config = json.load(f)
    threshold = config["threshold"]
    assert 0 < threshold < 1