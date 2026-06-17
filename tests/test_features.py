import pandas as pd
import numpy as np
from src.feature_engineering import engineer_features

def test_thermal_diff_calculation():
    # Setup dummy data
    data = pd.DataFrame({
        'Process temperature [K]': [308.6],
        'Air temperature [K]': [298.1],
        'Rotational speed [rpm]': [1500],
        'Torque [Nm]': [40],
        'Tool wear [min]': [10],
        'Type': ['M']
    })
    
    # Run function
    result = engineer_features(data)
    
    # Assert physical logic is correct
    assert result['Thermal_Diff'].iloc[0] == 308.6 - 298.1
    assert result['Frictional_Wear_Index'].iloc[0] == 40 * 10
    assert result['Type_Quality'].iloc[0] == 1 # 'M' mapped to 1