"""Mental tiredness prediction dashboard (educational prototype)."""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Mental Tiredness Prediction",
    page_icon="🧠",
    layout="centered",
)

MODEL_PATH = Path(__file__).with_name("app.joblib")
LOG_PATH = Path(__file__).with_name("prediction_logs.csv")
FEATURES = ["age", "sleep_hours", "study_hours", "stress_level"]
MODEL_VERSION = "demo-rf-v1.0.0"


def build_demo_model() -> dict:
    """Build a reproducible demonstration model when no trained artifact exists.

    Replace this synthetic-data step with a validated labelled dataset before any
    real-world use.
    """
    random = np.random.default_rng(42)
    rows = 1_200
    data = pd.DataFrame(
        {
            "age": random.integers(16, 61, rows),
            "sleep_hours": random.uniform(3.5, 10.0, rows),
            "study_hours": random.uniform(0.0, 14.0, rows),
            "stress_level": random.integers(1, 11, rows),
        }
    )
    # Synthetic target: 0 = low tiredness, 100 = severe tiredness.
    target = np.clip(
        18 + (10 - data["sleep_hours"]) * 7 + data["study_hours"] * 2.2
        + data["stress_level"] * 4.6 + np.maximum(data["age"] - 35, 0) * 0.25
        + random.normal(0, 5, rows),
        0,
        100,
    )
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "model": model,
        "features": FEATURES,
        "version": MODEL_VERSION,
        "algorithm": "Random Forest Regressor",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_source": "Synthetic demonstration data (1,200 records)",
        "metrics": {"mae": round(float(mean_absolute_error(y_test, predictions)), 2), "r2": round(float(r2_score(y_test, predictions)), 3)},
    }


@st.cache_resource
def load_model() -> dict:
    try:
        artifact = joblib.load(MODEL_PATH)
        if not isinstance(artifact, dict) or "model" not in artifact or artifact.get("version") != MODEL_VERSION:
            raise ValueError("Model artifact is missing or outdated")
        return artifact
    except (EOFError, FileNotFoundError, ValueError, OSError):
        artifact = build_demo_model()
        joblib.dump(artifact, MODEL_PATH)
        return artifact


def tiredness_band(value: float) -> tuple[str, str]:
    if value < 35:
        return "Low", "success"
    if value < 65:
        return "Moderate", "warning"
    return "High", "error"


artifact = load_model()
prediction_tab, mlops_tab = st.tabs(["Prediction", "MLOps overview"])

with prediction_tab:
    st.title("🧠 Mental Tiredness Prediction")
    st.write("Enter the details below to predict a mental tiredness score.")
    st.info("Educational demonstration only — not a medical diagnosis. Do not use this result as clinical advice.")

    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Age", min_value=13, max_value=100, value=22, step=1)
            study_hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
        with right:
            sleep_hours = st.number_input("Average sleep per night (hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            stress_level = st.slider("Current stress level", min_value=1, max_value=10, value=5, help="1 = very low stress; 10 = extremely high stress")
        submitted = st.form_submit_button("Predict mental tiredness", type="primary")

    if submitted:
        try:
            input_data = pd.DataFrame([[age, sleep_hours, study_hours, stress_level]], columns=FEATURES)
            result = float(artifact["model"].predict(input_data)[0])
            level, message_type = tiredness_band(result)

            st.subheader("Prediction result")
            getattr(st, message_type)(f"**{level} mental tiredness** — predicted score: **{result:.1f}/100**")
            st.caption("0 represents lower predicted tiredness and 100 represents higher predicted tiredness.")

            # Log each prediction locally for MLOps monitoring and later analysis.
            log_data = input_data.copy()
            log_data["prediction"] = round(result, 2)
            log_data["prediction_band"] = level
            log_data["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            log_data.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)
            st.info("Prediction logged successfully.")

            st.subheader("Input Details")
            st.dataframe(
                input_data.rename(
                    columns={"age": "Age", "sleep_hours": "Sleep hours", "study_hours": "Study hours", "stress_level": "Stress level"}
                ),
                hide_index=True,
                use_container_width=True,
            )
        except Exception as error:
            st.error("Prediction failed.")
            st.write("Check that the input column names match the columns used during model training.")
            st.exception(error)

    st.divider()
    st.caption("If tiredness or stress is persistent, severe, or affecting daily life, consider speaking with a qualified healthcare professional.")

with mlops_tab:
    st.header("MLOps overview")
    st.write("This tab makes the local model artifact, version, inputs, and evaluation information visible for demonstration and monitoring.")
    metrics = artifact["metrics"]
    first, second, third = st.columns(3)
    first.metric("Model", artifact["algorithm"])
    second.metric("Validation MAE", metrics["mae"])
    third.metric("Validation R²", metrics["r2"])
    st.markdown("**Recommended algorithm:** Random Forest Regressor is a suitable starting point for this small, tabular prediction problem because it captures non-linear relationships between sleep, study hours, stress, and tiredness without extensive feature scaling.")
    st.caption("For a production model, compare it against linear regression and gradient boosting using a real, consented, labelled dataset; select the model using held-out performance, calibration, fairness checks, and clinical review.")
    st.subheader("Model artifact")
    st.json(
        {
            "version": artifact["version"],
            "algorithm": artifact["algorithm"],
            "features": artifact["features"],
            "created_at_utc": artifact["created_at"],
            "training_source": artifact["training_source"],
            "artifact_path": str(MODEL_PATH),
            "prediction_log_path": str(LOG_PATH),
        }
    )
