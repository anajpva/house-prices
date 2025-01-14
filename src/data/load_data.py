import pandas as pd
import sys
import os

# Add project root to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

def load_data(filepath):
    """
    Carga un archivo CSV en un DataFrame de pandas.

    Args:
        filepath (str): Ruta al archivo CSV.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    """
    try:
        data_path = '../' + filepath
        data = pd.read_csv(data_path)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"FIle not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading file: {e}")
