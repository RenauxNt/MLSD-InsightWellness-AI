import requests

from insightwellness_ai.api.schema import MTRANS_FEATURES
from insightwellness_ai.config import API_BASE_URL

PREDICT_URL = f"{API_BASE_URL}/predict"
EXPLAIN_URL = f"{API_BASE_URL}/explain"

_session = requests.Session()

FEATURE_LABELS = {
    "Gender": "Gender",
    "Age": "Age",
    "family_history_with_overweight": "Family history of overweight",
    "FAVC": "Frequent high-caloric food",
    "FCVC": "Vegetable consumption",
    "NCP": "Main meals per day",
    "CAEC": "Eats between meals",
    "SMOKE": "Smokes",
    "CH2O": "Daily water intake",
    "SCC": "Monitors calorie intake",
    "FAF": "Physical activity frequency",
    "TUE": "Time on tech devices",
    "CALC": "Alcohol consumption",
    "MTRANS_automobile": "Transport: automobile",
    "MTRANS_motorbike": "Transport: motorbike",
    "MTRANS_bike": "Transport: bike",
    "MTRANS_walking": "Transport: walking",
}


def _normalize_binary(value):
    if isinstance(value, str):
        return 1 if value.lower() in ["yes", "true", "female"] else 0
    return int(value)


def _build_payload(
    Age: float,
    Gender: int,
    family_history_with_overweight: int,
    FAVC: int,
    FCVC: int,
    NCP: float,
    CAEC: int,
    SMOKE: int,
    CH2O: float,
    SCC: int,
    FAF: float,
    TUE: int,
    CALC: int,
    MTRANS: str,
) -> dict:
    data = {}
    data.update(
        {
            "Age": Age,
            "Gender": _normalize_binary(Gender),
            "family_history_with_overweight": _normalize_binary(
                family_history_with_overweight
            ),
            "FAVC": _normalize_binary(FAVC),
            "FCVC": FCVC,
            "NCP": NCP,
            "CAEC": CAEC,
            "SMOKE": _normalize_binary(SMOKE),
            "CH2O": CH2O,
            "SCC": _normalize_binary(SCC),
            "FAF": FAF,
            "TUE": TUE,
            "CALC": CALC,
        }
    )

    mtrans_map = {col.removeprefix("MTRANS_"): col for col in MTRANS_FEATURES}
    for key in mtrans_map.values():
        data[key] = 0
    if isinstance(MTRANS, str):
        data[mtrans_map.get(MTRANS.lower(), "MTRANS_automobile")] = 1
    return data


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

    data = _build_payload(
        Age,
        Gender,
        family_history_with_overweight,
        FAVC,
        FCVC,
        NCP,
        CAEC,
        SMOKE,
        CH2O,
        SCC,
        FAF,
        TUE,
        CALC,
        MTRANS,
    )

    try:
        response = _session.post(PREDICT_URL, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return f"Predicted obesity level: {result.get('prediction', 'unknown')}"
    except Exception as e:
        return f"Prediction error: {e}"


def explain_obesity_risk(
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
    Explain an obesity prediction using SHAP impact scores from the ML model.

    Use this whenever the user asks WHY a prediction was made, or asks
    which lifestyle factors are driving their predicted obesity class.
    Returns the predicted class plus the top 5 features ranked by SHAP
    impact, with the direction each one pushes (toward or away from the
    predicted class).

    Inputs match predict_obesity_risk. Missing fields use baseline defaults.
    """

    print("EXPLAIN TOOL CALLED")

    data = _build_payload(
        Age,
        Gender,
        family_history_with_overweight,
        FAVC,
        FCVC,
        NCP,
        CAEC,
        SMOKE,
        CH2O,
        SCC,
        FAF,
        TUE,
        CALC,
        MTRANS,
    )

    try:
        response = _session.post(
            EXPLAIN_URL, json=data, params={"top_n": 5}, timeout=60
        )
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        return f"Explain error: {e}"

    prediction = result.get("prediction", "unknown")
    drivers = result.get("explanation", {}).get("top_drivers", [])
    if not drivers:
        return f"Predicted obesity level: {prediction}. No SHAP drivers returned."

    lines = [f"Predicted obesity level: {prediction}.", "Top SHAP drivers:"]
    for d in drivers:
        feat = d.get("feature", "?")
        score = d.get("impact_score", 0.0)
        direction = "pushes toward" if score > 0 else "pushes away"
        label = FEATURE_LABELS.get(feat, feat)
        lines.append(f"- {label}: {direction} (impact={score:+.3f})")
    return "\n".join(lines)
