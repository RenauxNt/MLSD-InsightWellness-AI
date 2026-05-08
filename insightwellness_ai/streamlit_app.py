"""Streamlit dashboard for InsightWellness AI.

Two pages:
- Prediction: collects the 17 lifestyle features and calls the deployed
  Cloud Run API (/predict, /explain) to get a class + SHAP explanation.
- Data exploration: shows distributions of each feature and overlays
  the user's last submitted values for context.

Run locally with:
    streamlit run insightwellness_ai/streamlit_app.py
"""

import os

import gcsfs
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

DATA_PATH = "gs://mlops-2026-ramzan1/preprocessed/data_preprocessed.csv"
DEFAULT_API_URL = os.environ.get(
    "INSIGHTWELLNESS_API_URL",
    "https://insightwellness-api-545205658175.europe-west1.run.app",
)

CLASS_MAPPING = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III",
}

FEATURE_ORDER = [
    "Gender",
    "Age",
    "family_history_with_overweight",
    "FAVC",
    "FCVC",
    "NCP",
    "CAEC",
    "SMOKE",
    "CH2O",
    "SCC",
    "FAF",
    "TUE",
    "CALC",
    "MTRANS_automobile",
    "MTRANS_motorbike",
    "MTRANS_bike",
    "MTRANS_walking",
]

FEATURE_LABELS = {
    "Gender": "Gender",
    "Age": "Age (years)",
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
    "MTRANS_automobile": "Transport: Automobile",
    "MTRANS_motorbike": "Transport: Motorbike",
    "MTRANS_bike": "Transport: Bike",
    "MTRANS_walking": "Transport: Walking",
}

CATEGORICAL_OPTIONS = {
    "Gender": {0: "Male", 1: "Female"},
    "family_history_with_overweight": {0: "No", 1: "Yes"},
    "FAVC": {0: "No", 1: "Yes"},
    "FCVC": {1: "Never", 2: "Sometimes", 3: "Always"},
    "NCP": {1: "1", 2: "2", 3: "3", 4: "4+"},
    "CAEC": {0: "No", 1: "Sometimes", 2: "Frequently", 3: "Always"},
    "SMOKE": {0: "No", 1: "Yes"},
    "CH2O": {1: "<1L", 2: "1-2L", 3: ">2L"},
    "SCC": {0: "No", 1: "Yes"},
    "FAF": {0: "None", 1: "1-2 days", 2: "2-4 days", 3: "4-5 days"},
    "TUE": {0: "0-2h", 1: "3-5h", 2: ">5h"},
    "CALC": {0: "No", 1: "Sometimes", 2: "Frequently", 3: "Always"},
}

MTRANS_FEATURES = [f for f in FEATURE_ORDER if f.startswith("MTRANS_")]
MTRANS_LABEL_TO_KEY = {
    "Public transportation": None,
    "Automobile": "MTRANS_automobile",
    "Motorbike": "MTRANS_motorbike",
    "Bike": "MTRANS_bike",
    "Walking": "MTRANS_walking",
}
BINARY_FEATURES = {
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "SMOKE",
    "SCC",
    *MTRANS_FEATURES,
}


@st.cache_data(show_spinner="Loading dataset from GCS...")
def load_dataset(path: str) -> pd.DataFrame:
    fs = gcsfs.GCSFileSystem()
    with fs.open(path, "rb") as f:
        return pd.read_csv(f)


def call_api(endpoint: str, payload: dict, params: dict | None = None) -> dict:
    url = f"{DEFAULT_API_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    response = requests.post(url, json=payload, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def render_sidebar() -> str:
    st.sidebar.title("InsightWellness AI")
    st.sidebar.caption("Obesity-risk early warning")

    return st.sidebar.radio(
        "Page", ["Prediction", "Data exploration"], key="page"
    )


def render_feature_input(feature: str):
    label = FEATURE_LABELS[feature]
    if feature == "Age":
        return float(
            st.number_input(
                label,
                min_value=10.0,
                max_value=100.0,
                value=25.0,
                step=1.0,
                key=f"in_{feature}",
            )
        )
    options = CATEGORICAL_OPTIONS[feature]
    return st.selectbox(
        label,
        options=list(options.keys()),
        format_func=lambda v: options[v],
        key=f"in_{feature}",
    )


def render_input_form() -> dict:
    inputs: dict = {}
    left, right = st.columns(2)

    with left:
        st.subheader("Demographics & lifestyle")
        for feature in [
            "Gender",
            "Age",
            "family_history_with_overweight",
            "FAVC",
            "FCVC",
            "NCP",
            "CAEC",
            "SMOKE",
        ]:
            inputs[feature] = render_feature_input(feature)

    with right:
        st.subheader("Habits & transport")
        for feature in ["CH2O", "SCC", "FAF", "TUE", "CALC"]:
            inputs[feature] = render_feature_input(feature)

        choice = st.radio(
            "Main transport",
            list(MTRANS_LABEL_TO_KEY.keys()),
            key="mtrans_choice",
        )
        chosen_key = MTRANS_LABEL_TO_KEY[choice]
        for f in MTRANS_FEATURES:
            inputs[f] = 1 if f == chosen_key else 0

    return inputs


def render_probabilities(probabilities: dict) -> None:
    proba_df = (
        pd.DataFrame(
            {"class": list(probabilities.keys()), "probability": list(probabilities.values())}
        )
        .sort_values("probability", ascending=True)
    )
    fig = px.bar(
        proba_df,
        x="probability",
        y="class",
        orientation="h",
        title="Class probabilities",
        range_x=[0, 1],
    )
    st.plotly_chart(fig, width="stretch")


def render_shap_drivers(drivers: list[dict]) -> None:
    drivers_df = pd.DataFrame(drivers)
    drivers_df["direction"] = drivers_df["impact_score"].apply(
        lambda v: "pushes toward" if v > 0 else "pushes away"
    )
    fig = px.bar(
        drivers_df.iloc[::-1],
        x="impact_score",
        y="feature",
        orientation="h",
        color="direction",
        title=f"Top {len(drivers_df)} SHAP drivers for the predicted class",
        color_discrete_map={"pushes toward": "#e6550d", "pushes away": "#3182bd"},
    )
    st.plotly_chart(fig, width="stretch")


def page_prediction() -> None:
    st.title("Obesity-risk prediction")
    st.write(
        "Fill in the lifestyle features below. The app sends the data to the "
        "deployed InsightWellness API and returns a predicted obesity class "
        "with a SHAP explanation."
    )

    inputs = render_input_form()
    st.session_state["last_inputs"] = inputs

    cols = st.columns([1, 1, 4])
    with cols[0]:
        run_predict = st.button("Predict", type="primary")
    with cols[1]:
        run_explain = st.button("Predict + Explain (SHAP)")

    if run_predict:
        try:
            result = call_api("predict", inputs)
            st.success(f"Predicted class: **{result['prediction']}**")
            with st.expander("Raw response"):
                st.json(result)
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")

    if run_explain:
        try:
            result = call_api("explain", inputs, params={"top_n": 8})
            st.success(f"Predicted class: **{result['prediction']}**")
            if result.get("probabilities"):
                render_probabilities(result["probabilities"])
            drivers = result.get("explanation", {}).get("top_drivers", [])
            if drivers:
                render_shap_drivers(drivers)
            with st.expander("Raw response"):
                st.json(result)
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")


def build_distribution_plot(
    df: pd.DataFrame, feature: str, user_value, template: str
):
    is_categorical = feature in BINARY_FEATURES
    label_map = CATEGORICAL_OPTIONS.get(feature)

    if is_categorical:
        counts = df[feature].value_counts().sort_index().reset_index()
        counts.columns = [feature, "count"]
        counts["label"] = (
            counts[feature].map(label_map).fillna(counts[feature].astype(str))
            if label_map is not None
            else counts[feature].astype(str)
        )
        fig = px.bar(
            counts,
            x="label",
            y="count",
            template=template,
            title=f"Distribution of {FEATURE_LABELS.get(feature, feature)}",
            labels={"label": feature},
        )
        if user_value is not None:
            user_label = (
                label_map.get(user_value, str(user_value))
                if label_map
                else str(user_value)
            )
            colors = [
                "#e6550d" if str(x) == user_label else "#1f77b4"
                for x in counts["label"]
            ]
            fig.update_traces(marker_color=colors)
            fig.add_annotation(
                text=f"You selected: {user_label}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.08,
                showarrow=False,
                font=dict(color="#e6550d"),
            )
    else:
        fig = px.histogram(
            df,
            x=feature,
            nbins=40,
            template=template,
            title=f"Distribution of {FEATURE_LABELS.get(feature, feature)}",
        )
        if user_value is not None:
            fig.add_vline(
                x=user_value,
                line_color="#e6550d",
                line_width=2,
                line_dash="dash",
                annotation_text=f"You: {user_value:g}",
                annotation_position="top",
            )
    return fig


def page_exploration() -> None:
    st.title("Data exploration")
    df = load_dataset(DATA_PATH)
    st.caption(
        f"Loaded {len(df):,} preprocessed records with {df.shape[1]} columns "
        f"from `{DATA_PATH}`."
    )

    template = "plotly_white"
    user_inputs = st.session_state.get("last_inputs")

    with st.expander("Preview the dataset"):
        st.dataframe(df.head(50))

    st.subheader("Target distribution")
    class_counts = df["Obesity"].map(CLASS_MAPPING).value_counts().reset_index()
    class_counts.columns = ["class", "count"]
    class_order = [CLASS_MAPPING[i] for i in sorted(CLASS_MAPPING)]
    fig = px.bar(
        class_counts,
        x="class",
        y="count",
        template=template,
        title="Obesity class balance",
        category_orders={"class": class_order},
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Feature distributions")
    if user_inputs is None:
        st.info(
            "Submit a prediction first to overlay your value on each distribution."
        )
    feature = st.selectbox(
        "Pick a feature",
        [f for f in FEATURE_ORDER if f in df.columns],
        format_func=lambda f: FEATURE_LABELS.get(f, f),
    )
    user_value = user_inputs.get(feature) if user_inputs else None
    st.plotly_chart(
        build_distribution_plot(df, feature, user_value, template),
        width="stretch",
    )

    st.subheader("Correlation heatmap")
    corr_cols = [f for f in FEATURE_ORDER + ["Obesity"] if f in df.columns]
    fig = px.imshow(
        df[corr_cols].corr(),
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        template=template,
        title="Pearson correlation",
    )
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="InsightWellness AI",
        layout="wide",
    )
    page = render_sidebar()
    if page == "Prediction":
        page_prediction()
    else:
        page_exploration()


if __name__ == "__main__":
    main()
