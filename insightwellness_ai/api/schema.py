"""Model input schema and output class mapping shared by the API routes."""

EXPECTED_MODEL_SCHEMA = {
    "Gender": {"description": "Gender (0 = Male, 1 = Female)", "valid_values": [0, 1]},
    "Age": {"description": "Age in years", "valid_values": "Numeric (e.g., 25.0)"},
    "family_history_with_overweight": {
        "description": "Family history of overweight (0 = False, 1 = True)",
        "valid_values": [0, 1],
    },
    "FAVC": {
        "description": "Frequent consumption of high caloric food (0 = False, 1 = True)",
        "valid_values": [0, 1],
    },
    "FCVC": {
        "description": "Frequency of consumption of vegetables (1 = Never, 2 = Sometimes, 3 = Always)",
        "valid_values": [1, 2, 3],
    },
    "NCP": {
        "description": "Number of main meals per day (1, 2, 3, or 4+ meals)",
        "valid_values": [1, 2, 3, 4],
    },
    "CAEC": {
        "description": "Consumption of food between meals (0=No, 1=Sometimes, 2=Frequently, 3=Always)",
        "valid_values": [0, 1, 2, 3],
    },
    "SMOKE": {
        "description": "Does the patient smoke? (0 = False, 1 = True)",
        "valid_values": [0, 1],
    },
    "CH2O": {
        "description": "Consumption of water daily (1 = Less than 1L, 2 = 1 to 2L, 3 = More than 2L)",
        "valid_values": [1, 2, 3],
    },
    "SCC": {
        "description": "Calories consumption monitoring (0 = False, 1 = True)",
        "valid_values": [0, 1],
    },
    "FAF": {
        "description": "Physical activity frequency (0 = None, 1 = 1 to 2 days, 2 = 2 to 4 days, 3 = 4 to 5 days)",
        "valid_values": [0, 1, 2, 3],
    },
    "TUE": {
        "description": "Time using technology devices (0 = 0 to 2 hours, 1 = 3 to 5 hours, 2 = More than 5 hours)",
        "valid_values": [0, 1, 2],
    },
    "CALC": {
        "description": "Consumption of alcohol (0=No, 1=Sometimes, 2=Frequently, 3=Always)",
        "valid_values": [0, 1, 2, 3],
    },
    "MTRANS_automobile": {
        "description": "Main transport is automobile (0=No, 1=Yes). Max ONE MTRANS variable can be 1.",
        "valid_values": [0, 1],
    },
    "MTRANS_motorbike": {
        "description": "Main transport is motorbike (0=No, 1=Yes). Max ONE MTRANS variable can be 1",
        "valid_values": [0, 1],
    },
    "MTRANS_bike": {
        "description": "Main transport is bike (0=No, 1=Yes). Max ONE MTRANS variable can be 1.",
        "valid_values": [0, 1],
    },
    "MTRANS_walking": {
        "description": "Main transport is walking (0=No, 1=Yes). Max ONE MTRANS variable can be 1.",
        "valid_values": [0, 1],
    },
}

# Translates the model's numeric output back to readable labels
CLASS_MAPPING = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III",
}
