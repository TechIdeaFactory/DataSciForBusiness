# pip install streamlit
# pip install pandas
# pip install requests
# pip install -U scikit-learn
# pip install joblib
# needs random_forest_model.pkl
# streamlit run streamlit_app.py

import streamlit as st
import joblib
from typing import Tuple
import requests
# import pandas as pd

LOCAL_MODEL = "Local"
REST_API_MODEL = "REST API"
NO_OPTION = "N"
YES_OPTION = "Y"


@st.cache_resource
def load_model():
    # Load the saved
    # Random Forest model
    rf_model = joblib.load(
        "random_forest_model.pkl"
    )
    return rf_model


# Initialize session
# data
def initialize_session_state():
    if (
        "ever_smoked"
        not in st.session_state
    ):
        st.session_state.ever_smoked = (
            NO_OPTION
        )


# Format features
# for model
def format_model_inputs(
        bmi: int,
        hbA1c: float,
        current_smoker: str,
        ever_smoked: str,
) -> Tuple[int, float, int, int]:
    current_smoker_flag = 0
    never_smoked_flag = 1

    if current_smoker == YES_OPTION:
        current_smoker_flag = 1

    if ever_smoked == YES_OPTION:
        never_smoked_flag = 0

    return (
        int(bmi),
        float(hbA1c),
        current_smoker_flag,
        never_smoked_flag,
    )


# Predict using
# model and inputs
def model_predict(
    model: str,
    bmi: int,
    hbA1c: float,
    current_smoker: str,
    ever_smoked: str,
) -> str:

    (
        bmi_input,
        hbA1c_input,
        current_smoker_input,
        never_smoked_input,
    ) = format_model_inputs(
        bmi,
        hbA1c,
        current_smoker,
        ever_smoked,
    )

    model_pred = 0

    if model == LOCAL_MODEL:
        # Load the model
        rf_model = load_model()

        # Make a prediction
        # for a single
        # data point
        d = {
            "bmi": [bmi_input],
            "HbA1c_level": [
                hbA1c_input
            ],
            "smoking_history_current": [
                current_smoker_input
            ],
            "smoking_history_never": [
                never_smoked_input
            ],
        }
        # df = pd.DataFrame(data=d)
        # print(df)
        # print(rf_model)
        # model_pred = rf_model.predict(
        #    df
        # ).item(0)

    if model_pred == 1:
        return "Diabetic"
    else:
        return "Non Diabetic"


# Initialize session state
initialize_session_state()

header = st.container(border=True)
model = st.container(border=True)
predict = st.container(border=True)

# Create header
# with centered text
# section
with header:
    st.markdown(
        "<h1 style='text-align: center;'>"
        "Diabetes ML Model App</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='text-align: center;'>"
        "Predicts whether person has diabetes</div>",
        unsafe_allow_html=True,
    )

# Create model
# with inputs
# section
with model:
    # Create two columns
    # within the container
    (
        col1,
        col2,
        col3,
        col4,
    ) = model.columns(4)

    # Dictionary containing
    # options for selection
    smoking_options_dict = {
        NO_OPTION: [
            NO_OPTION,
            YES_OPTION,
        ],
        YES_OPTION: [YES_OPTION],
    }

    # Create a current smoker
    # dropdown box
    current_smoker = col1.selectbox(
        "Current Smoker:",
        smoking_options_dict[NO_OPTION],
        index=0,
    )

    # Create a ever smoked
    # dropdown box
    index = 0
    if st.session_state.ever_smoked:
        if (
            st.session_state.ever_smoked
            == YES_OPTION
            and current_smoker
            == NO_OPTION
        ):
            index = 1

    ever_smoked = col3.selectbox(
        "Ever Smoked:",
        smoking_options_dict[
            current_smoker
        ],
        index=index,
    )

    # Update session state
    st.session_state.ever_smoked = (
        ever_smoked
    )

    # Create a HbA1c_level
    # input
    hbA1c_level = col1.text_input(
        "Enter HbA1c Level:", ""
    )

    # Create a BMI
    # slider
    bmi = col3.slider(
        "BMI",
        min_value=10.0,
        max_value=40.0,
        value=20.0,
        step=1.0,
    )


# Create a
# prediction section
with predict:
    # Create two columns
    # within the container
    (
        predict_col1,
        predict_col2,
        predict_col3,
        predict_col4,
    ) = predict.columns(4)

    # Create a model
    # LOCAL / REST API
    # dropdown box
    model = predict_col1.selectbox(
        "Model:",
        [LOCAL_MODEL, REST_API_MODEL],
        index=0,
    )

    # Create a predict button
    predict_clicked = (
        predict_col3.button("Predict")
    )

    # Check if the
    # predict button is clicked
    if predict_clicked:
        prediction = model_predict(
            model,
            bmi,
            hbA1c_level,
            current_smoker,
            ever_smoked
        )
        predict_col4.write(prediction)
