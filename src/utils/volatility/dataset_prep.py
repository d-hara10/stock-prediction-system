import pandas as pd
# Prepare Features & Target
def prepare_X_y(data: pd.DataFrame, feature_list):
    X = data[feature_list]
    y = data["TargetVolatility"]
    return X, y