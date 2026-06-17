import pandas as pd
import numpy as np

def load_data(filepath):
    return pd.read_csv(filepath)

def engineer_features(df):
    X =df.copy()
    X = X.dropna()
    X['Thermal_Diff'] = X['Process temperature [K]'] - X['Air temperature [K]']
    X['Mechanical_Power'] = (2 * np.pi * X['Rotational speed [rpm]'] * X['Torque [Nm]']) / 60
    X['Frictional_Wear_Index'] = X['Torque [Nm]'] * X['Tool wear [min]']

    quality_mapping =  {'L':0, 'M': 1, 'H':2}
    X['Type_Quality'] =  X['Type'].map(quality_mapping)

    cols_to_drop = [
        'UDI',
        'Product ID',
        'Type'
    ]
    X= X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')
    return X
