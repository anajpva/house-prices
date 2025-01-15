import pandas as pd

def load_data(filepath):
    # Load CSV file as a pandas DataFrame
    try:
        data_path = '../' + filepath
        data = pd.read_csv(data_path)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"FIle not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading file: {e}")
