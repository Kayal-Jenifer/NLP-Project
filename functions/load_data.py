import pandas as pd
from .divider import divider
import os


def load_data() -> pd.DataFrame:
    divider("INITIALIZATION: LOAD DATA")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "data", "Industrial_and_Scientific.json")
    df = pd.read_json(filepath, lines=True)
    print(f"Dataset loaded successfully!")
    return df
