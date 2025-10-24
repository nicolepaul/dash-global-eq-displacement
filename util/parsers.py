import pandas as pd
from _config import *


def get_data():

    # Read data
    data = pd.read_csv(PATH_DATA)

    # Apply some transformations
    data["damaged*ahhs"] = data["damaged"] * data["ahhs"]
    data["destroyed*ahhs-fatalities"] = (
        data["destroyed"] * data["ahhs"] - data["fatalities"]
    )
    data["(damaged+destroyed)*ahhs-fatalities"] = (
        data["damaged"] + data["destroyed"]
    ) * data["ahhs"] - data["fatalities"]

    # Duplicate for drivers logic
    data["DAMAGED"] = data["damaged*ahhs"]
    data["DESTROYED"] = data["(damaged+destroyed)*ahhs-fatalities"]

    # Hard encode some values
    factors = {
        "damaged*ahhs": "Residents of damaged dwellings",
        "destroyed*ahhs-fatalities": "Residents of destroyed dwellings",
        "(damaged+destroyed)*ahhs-fatalities": "Residents of damaged + destroyed dwellings",
    }
    metrics = {
        "evacuated": "Evacuated (peak)",
        "sheltered_peak": "Sheltered (peak)",
        "protracted": "Protracted (~6mo)",
        "assisted": "Assisted",
    }

    categories = {"region": "Geographical region", "income": "Country income level"}
    data["income"] = pd.Categorical(
        data["income"],
        categories=[
            "Low income",
            "Lower middle income",
            "Upper middle income",
            "High income",
        ],
        ordered=True,
    )

    return data, factors, metrics, categories


def get_drivers():

    drivers = pd.read_csv(PATH_DRIVERS)  # .sort_values(by=['category', 'name'])

    return drivers
