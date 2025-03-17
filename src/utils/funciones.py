import numpy as np


month_to_day = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

# Función para convertir el mes de texto a número

def convert_month_to_day(X):
    return np.array([[month_to_day[m]] for m in X.values.ravel()])


def cat_binary(df):
    df["default"] = df["default"] == "yes"
    df["housing"] = df["housing"] == "yes"
    df["loan"] = df["loan"] == "yes"
    return df