import requests

API_URL = "https://insightwellness-api-545205658175.europe-west1.run.app/predict"

# Default valid baseline input
DEFAULT_INPUT = {
    "Gender": 0,
    "Age": 25,
    "family_history_with_overweight": 0,
    "FAVC": 0,
    "FCVC": 2,
    "NCP": 3,
    "CAEC": 1,
    "SMOKE": 0,
    "CH2O": 2,
    "SCC": 0,
    "FAF": 1,
    "TUE": 1,
    "CALC": 1,
    "MTRANS_automobile": 1,
    "MTRANS_motorbike": 0,
    "MTRANS_bike": 0,
    "MTRANS_walking": 0,
}


def _normalize_binary(value):
    if isinstance(value, str):
        return 1 if value.lower() in ["yes", "true", "female"] else 0
    return int(value)


def predict_obesity_risk(
    Age: float = 25,
    Gender: int = 0,
    family_history_with_overweight: int = 0,
    FAVC: int = 0,
    FCVC: int = 2,
    NCP: float = 3,
    CAEC: int = 1,
    SMOKE: int = 0,
    CH2O: float = 2,
    SCC: int = 0,
    FAF: float = 1,
    TUE: int = 1,
    CALC: int = 1,
    MTRANS: str = "automobile",
) -> str:
    """
    Predict obesity level using the trained ML model.

    MUST be used whenever the user provides ANY personal, dietary, or lifestyle information.

    Supported inputs include:
    - Demographics: Age, Gender
    - Family history
    - Eating habits (FAVC, FCVC, CAEC, NCP)
    - Lifestyle (FAF, SMOKE, CH2O, TUE, CALC)
    - Transportation (MTRANS: automobile, motorbike, bike, walking)

    The tool automatically fills missing values with defaults.

    Returns:
        Predicted obesity class.
    """

    print("ML TOOL CALLED")

    data = DEFAULT_INPUT.copy()

    Gender = _normalize_binary(Gender)
    family_history_with_overweight = _normalize_binary(family_history_with_overweight)
    FAVC = _normalize_binary(FAVC)
    SMOKE = _normalize_binary(SMOKE)
    SCC = _normalize_binary(SCC)

    data.update(
        {
            "Age": Age,
            "Gender": Gender,
            "family_history_with_overweight": family_history_with_overweight,
            "FAVC": FAVC,
            "FCVC": FCVC,
            "NCP": NCP,
            "CAEC": CAEC,
            "SMOKE": SMOKE,
            "CH2O": CH2O,
            "SCC": SCC,
            "FAF": FAF,
            "TUE": TUE,
            "CALC": CALC,
        }
    )

    mtrans_map = {
        "automobile": "MTRANS_automobile",
        "motorbike": "MTRANS_motorbike",
        "bike": "MTRANS_bike",
        "walking": "MTRANS_walking",
    }

    for key in [
        "MTRANS_automobile",
        "MTRANS_motorbike",
        "MTRANS_bike",
        "MTRANS_walking",
    ]:
        data[key] = 0

    if isinstance(MTRANS, str):
        key = mtrans_map.get(MTRANS.lower(), "MTRANS_automobile")
        data[key] = 1

    try:
        response = requests.post(API_URL, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()

        return f"Predicted obesity level: {result.get('prediction', 'unknown')}"

    except Exception as e:
        return f"Prediction error: {e}"
