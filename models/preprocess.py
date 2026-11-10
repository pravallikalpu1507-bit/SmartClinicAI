import pandas as pd

def load_data():
    """Load the heart disease structured dataset."""
    data = pd.read_csv("Data/heart.csv")
    return data

def clean_data(data):
    """Clean and preprocess the dataset."""
    print(f"Before cleaning: {data.shape}")
    data = data.dropna()
    data = data.drop_duplicates()
    print(f"After cleaning:  {data.shape}")
    return data
