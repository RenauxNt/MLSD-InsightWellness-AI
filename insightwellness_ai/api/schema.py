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

MTRANS_FEATURES = [f for f in EXPECTED_MODEL_SCHEMA if f.startswith("MTRANS_")]

# CLASS_MAPPING and the pipeline's CSV mapping both derive from this
CLASS_LABELS = (
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
)

CLASS_MAPPING = dict(enumerate(CLASS_LABELS))


def patient_data_definition() -> dict:
    """Swagger 'PatientData' body definition, generated from the schema."""
    properties = {}
    for name, spec in EXPECTED_MODEL_SCHEMA.items():
        valid_values = spec["valid_values"]
        if isinstance(valid_values, list):
            properties[name] = {
                "type": "integer",
                "enum": valid_values,
                "description": spec["description"],
                # first valid value keeps the example payload itself valid
                "example": valid_values[0],
            }
        else:
            properties[name] = {
                "type": "number",
                "description": spec["description"],
                "example": 25.5,
            }
    return {
        "type": "object",
        "required": list(EXPECTED_MODEL_SCHEMA),
        "properties": properties,
    }
