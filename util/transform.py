import re
from itertools import combinations

def add_threshold_indicator(df, col, threshold):
    name = f"I({col}>{threshold})"
    df[name] = (df[col] > threshold).astype(int)
    return df

def add_interaction(df, col1, col2):
    name = f"{col1}×{col2}"
    df[name] = df[col1] * df[col2]
    return df

def parse_transformations(df, features):

    # Model data
    model_data = df.copy()

    # Handle destroyed and damaged capitalization
    model_data.rename(columns={'damaged': 'DAMAGED', 'destroyed': 'DESTROYED'}, inplace=True)

    # Handle any indicator thresholds
    pattern = re.compile(r"^I\(([^>]+)>([^)]+)\)$")
    for feature in features:
        match = pattern.match(feature)
        if match:
            var, thresh = match.groups()
            try:
                thresh = int(thresh)
            except ValueError:
                thresh = float(thresh)
            model_data = add_threshold_indicator(model_data, var, thresh)
    
    # Handle any interactions
    pattern = re.compile(r"^([^×]+)×([^×]+)$")
    for feature in features:
        match = pattern.match(feature)
        if match:
            var1, var2 = match.groups()
            model_data = add_interaction(model_data, var1, var2)

    return model_data[features]


def generate_predictor_subsets(predictors):

    # Assume destroyed is always the first variable and mandatory
    base = predictors[0]
    optional = predictors[1:]
    
    # Find all subsets with optional variables
    subsets = []
    for r in range(len(optional) + 1):
        for combo in combinations(optional, r):
            subset = [base] + list(combo)
            subsets.append(subset)

    return subsets