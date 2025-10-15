import pandas as pd
from _config import *

def get_data():

    # Read data
    data = pd.read_csv(PATH_DATA)

    # Apply some transformations
    data["damaged*ahhs"] = data['damaged']*data['ahhs']
    data[ "destroyed*ahhs-fatalities"] = data['destroyed']*data['ahhs'] - data['fatalities']
    data["(damaged+destroyed)*ahhs-fatalities"] = (data['damaged']+data['destroyed'])*data['ahhs'] - data['fatalities']

    # Duplicate for drivers logic
    data['DAMAGED'] = data["damaged*ahhs"]
    data['DESTROYED'] = data["(damaged+destroyed)*ahhs-fatalities"]

    # Hard encode some values
    factors = {
        "damaged*ahhs": "Residents of damaged dwellings",
        "destroyed*ahhs-fatalities": "Residents of destroyed dwellings",
        "(damaged+destroyed)*ahhs-fatalities": "Residents of damaged + destroyed dwellings",
    }
    metrics = {
        "evacuated": "Evacuated (peak)",
        "sheltered_peak": "Sheltered (peak)",
        # "sheltered_>1m": "Sheltered (>1mo)",
        # "sheltered_>3m": "Sheltered (>3mo)",
        # "sheltered_>6m": "Sheltered (>6mo)",
        "needs": "Protracted (~6mo)",
        "assisted": "Assisted",
    }

    

    return data, factors, metrics

def get_drivers():

    drivers = pd.read_csv(PATH_DRIVERS)

    return drivers