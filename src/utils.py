
import numpy as np
from sklearn.metrics import mean_squared_error


def print_summary(data):
    """
    Imprime un resumen básico del dataset.

    Args:
        data (pd.DataFrame): Dataset a resumir.
    """
    print("Primeras filas del dataset:")
    print(data.head())
    print("\nResumen estadístico:")
    print(data.describe())
    print("\nInformación del dataset:")
    data.info()

def model_summary(model_name, model, X_test, y_test):
    model_best_rmse = np.sqrt(-1 * model.best_score_) 
    
    print(f"Best {model_name} MRSE result: {round(model_best_rmse, 4)}")

    model_best_model = model.best_estimator_
    y_pred_xgb = model_best_model.predict(X_test)

    model_mse = mean_squared_error(y_test, y_pred_xgb)
    model_rmse = np.sqrt(model_mse)
    
    print(f"Test {model_name} MRSE result: {round(model_rmse, 4)}")

    return model_rmse