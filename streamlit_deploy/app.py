from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(145deg, #f6fbff 0%, #eef4ff 55%, #f9f7ff 100%);
    }
    .block-container {max-width: 920px; padding-top: 2rem;}
    .hero {
        padding: 1.8rem 2rem;
        border-radius: 22px;
        color: white;
        background: linear-gradient(120deg, #0d3b66, #326fa8 58%, #7b61a8);
        box-shadow: 0 14px 35px rgba(13, 59, 102, 0.18);
        margin-bottom: 1.4rem;
    }
    .hero h1 {margin: 0; font-size: 2.15rem;}
    .hero p {margin: .55rem 0 0; opacity: .9; font-size: 1.02rem;}
    .result-card {
        padding: 1.3rem 1.5rem;
        border-radius: 18px;
        background: white;
        border: 1px solid #dce8f5;
        box-shadow: 0 8px 24px rgba(28, 64, 96, 0.10);
        margin-top: 1rem;
    }
    .survived {border-left: 7px solid #159957;}
    .not-survived {border-left: 7px solid #d64545;}
    .small-note {color: #5e6b78; font-size: .88rem;}
    div.stButton > button {
        width: 100%; border-radius: 12px; height: 3rem;
        font-weight: 700; background: #0d3b66; color: white; border: 0;
    }
    div.stButton > button:hover {background: #15558a; color: white;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🚢 Titanic Survival Predictor</h1>
      <p>Enter passenger details to estimate survival probability using machine-learning model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


MODEL_PATH = Path(__file__).with_name("titanic_best_model.pkl")


@st.cache_resource
def load_model_bundle(path: Path):
    """Load the trusted model bundle once for the Streamlit session."""
    with path.open("rb") as file:
        loaded = pickle.load(file)

    # Preferred format created by the notebook; direct estimators are tolerated.
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded

    feature_names = list(getattr(loaded, "feature_names_in_", []))
    return {
        "model": loaded,
        "model_name": type(loaded).__name__,
        "feature_names": feature_names,
        "target_mapping": {0: "Did not survive", 1: "Survived"},
    }


def make_model_input(
    feature_names, pclass, sex, age, sibsp, parch, fare, embarked
):
    """Convert friendly webpage inputs into the final_data dummy-column schema."""
    row = {str(feature): 0.0 for feature in feature_names}

    direct_values = {
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Pclass": pclass,
    }
    for column, value in direct_values.items():
        if column in row:
            row[column] = value

    # Handles both pd.get_dummies names (2, 3, male, Q, S) and prefixed names.
    dummy_values = {
        "2": int(pclass == 2),
        "3": int(pclass == 3),
        "Pclass_2": int(pclass == 2),
        "Pclass_3": int(pclass == 3),
        "male": int(sex == "Male"),
        "Sex_male": int(sex == "Male"),
        "Q": int(embarked == "Queenstown"),
        "S": int(embarked == "Southampton"),
        "Embarked_Q": int(embarked == "Queenstown"),
        "Embarked_S": int(embarked == "Southampton"),
    }
    for column, value in dummy_values.items():
        if column in row:
            row[column] = value

    return pd.DataFrame([row], columns=[str(name) for name in feature_names])


if not MODEL_PATH.exists():
    st.error("Model file not found: `titanic_best_model.pkl`")
    st.info(
        "Run the serialization cell in the notebook, then place the generated "
        "pickle file in the same folder as `app.py`."
    )
    st.stop()

try:
    bundle = load_model_bundle(MODEL_PATH)
except Exception as exc:
    st.error("The model could not be loaded.")
    st.caption(f"Technical detail: {exc}")
    st.stop()

model = bundle["model"]
feature_names = [str(name) for name in bundle.get("feature_names", [])]

if not feature_names:
    st.error("The pickle file does not contain the training feature names.")
    st.stop()

with st.form("passenger_form"):
    st.subheader("Passenger details")

    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox(
            "Passenger class",
            options=[1, 2, 3],
            format_func=lambda value: {1: "1st class", 2: "2nd class", 3: "3rd class"}[value],
        )
        sex = st.selectbox("Sex", ["Female", "Male"])
        age = st.number_input("Age", min_value=0.1, max_value=100.0, value=30.0, step=1.0)
        fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

    with col2:
        sibsp = st.number_input(
            "Siblings / spouses aboard", min_value=0, max_value=10, value=0, step=1
        )
        parch = st.number_input(
            "Parents / children aboard", min_value=0, max_value=10, value=0, step=1
        )
        embarked = st.selectbox(
            "Port of embarkation",
            ["Southampton", "Cherbourg", "Queenstown"],
        )

    submitted = st.form_submit_button("Predict survival")

if submitted:
    passenger = make_model_input(
        feature_names, pclass, sex, age, sibsp, parch, fare, embarked
    )

    try:
        probability = float(model.predict_proba(passenger)[0, 1])
        prediction = int(model.predict(passenger)[0])
    except Exception as exc:
        st.error("Prediction failed because the webpage inputs do not match the model schema.")
        st.caption(f"Technical detail: {exc}")
        st.stop()

    label = bundle.get("target_mapping", {0: "Did not survive", 1: "Survived"}).get(
        prediction, str(prediction)
    )
    card_class = "survived" if prediction == 1 else "not-survived"
    icon = "✅" if prediction == 1 else "⚠️"

    st.markdown(
        f"""
        <div class="result-card {card_class}">
          <h2>{icon} Prediction: {label}</h2>
          <p>The model estimates a <strong>{probability:.1%}</strong> probability of survival.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(int(np.clip(probability * 100, 0, 100)))
    metric1, metric2 = st.columns(2)
    metric1.metric("Survival probability", f"{probability:.1%}")
    metric2.metric("Model", bundle.get("model_name", type(model).__name__))

    with st.expander("View model-ready input"):
        st.dataframe(passenger, use_container_width=True, hide_index=True)

st.markdown(
    "<p class='small-note'>Educational demonstration only. This prediction describes a historical dataset and should not be used for real safety decisions.</p>",
    unsafe_allow_html=True,
)
