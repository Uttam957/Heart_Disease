import streamlit as st
import pandas as pd
import joblib

model=joblib.load("Logistic_regression_heart.pkl")
scaler=joblib.load("scaler.pkl")
expected_columns=joblib.load("columns.pkl")


st.title("Heart Stroke Prediction")
st.markdown("Provide the foloowing details")

age=st.slider("age",18,100,40)
sex=st.selectbox("Sex",['M','F'])
chest_pain=st.selectbox("Chest pain type", ["ATA","NAP","TA","ASY"])
resting_BP=st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol=st.number_input("Cholesterol (mm/dL)",100, 600, 200)
fasting_BS=st.selectbox("Fasting Blood Sugar > 120 mg/dL",[0,1])
resting_ECG=st.selectbox("Resting ECG",["Normal","ST","LVH"])
max_HR=st.slider("Max Heart Rate", 60,220,150)
exercise_angina=st.selectbox("Exercise-Induced Angina",["Y","N"])
oldPeak=st.slider("OldPeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope=st.selectbox("St Slope",["Up","Flat","Down"])


if st.button("Predict"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_BP,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_BS,
        'MaxHR': max_HR,
        'Oldpeak': oldPeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ECG: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_data=pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col]=0

    input_data=input_data[expected_columns]

    scaled_input=scaler.transform(input_data)
    prediction=model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️High Risk of Heart Disease")
    else:
        st.success("✅Low Risk of Heart Rate")    