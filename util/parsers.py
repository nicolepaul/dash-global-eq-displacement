import pandas as pd


def get_data():

    # Read data
    file_name = "data.csv"
    data = pd.read_csv(file_name)

    # Apply some transformations
    data["damaged*ahhs"] = data['damaged']*data['ahhs']
    data[ "destroyed*ahhs-fatalities"] = data['destroyed']*data['ahhs'] - data['fatalities']
    data["(damaged+destroyed)*ahhs-fatalities"] = (data['damaged']+data['destroyed'])*data['ahhs'] - data['fatalities']

    # Hard encode some values
    factors = {
        "damaged*ahhs": "Residents of damaged dwellings",
        "destroyed*ahhs-fatalities": "Residents of destroyed dwellings",
        "(damaged+destroyed)*ahhs-fatalities": "Residents of damaged + destroyed dwellings",
    }
    metrics = {
        "evacuated": "Evacuated (peak)",
        "sheltered_peak": "Sheltered (peak)",
        "sheltered_>1m": "Sheltered (>1mo)",
        "sheltered_>3m": "Sheltered (>3mo)",
        "sheltered_>6m": "Sheltered (>6mo)",
        "needs": "Needs (~6mo)",
        "assisted": "Assisted (~6mo)",
    }

    return data, factors, metrics
