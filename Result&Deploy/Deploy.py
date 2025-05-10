import streamlit as st
import pandas as pd
import os
import sys
base_path = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(base_path, '../Model')
sys.path.append(utils_path)

from predict import prediction

form_values = {
    "Target" : "Steroid-Induced Diabetes",
    "Genetic Markers" : "Positive",
    "Autoantibodies" : "Positive",
    "Family History" : "Yes",
    "Environmental Factors" : "Present",
    "Insulin Levels" : None,
    "Age" : None,
    "BMI" : None,
    "Physical Activity" : "High",
    "Dietary Habits" : "Healthy",
    "Blood Pressure" : None,
    "Cholesterol Levels" : None,
    "Waist Circumference" : None,
    "Blood Glucose Levels" : None,
    "Ethnicity" : "Low Risk",
    "Socioeconomic Factors" : "Low",
    "Smoking Status" : "Smoker",
    "Alcohol Consumption" : "High",
    "Glucose Tolerance Test" : "Normal",
    "History of PCOS" : "No",
    "Previous Gestational Diabetes" : "No",
    "Pregnancy History" : "Normal",
    "Weight Gain During Pregnancy" : None,
    "Pancreatic Health" : None,
    "Pulmonary Function" : None,
    "Cystic Fibrosis Diagnosis" : "No",
    "Steroid Use History" : "No",
    "Genetic Testing" : "Positive",
    "Neurological Assessments" : 1,
    "Liver Function Tests" : "Normal",
    "Digestive Enzyme Levels" : None,
    "Urine Test" : "Normal",
    "Birth Weight" : None,
    "Early Onset Symptoms" : "No"
}

st.title("Diabetes assessment and prediction.")

st.subheader("Please fill in your information in the form below:")

with st.form(key = "inform"):
    name = st.text_input("Full name: ")

    form_values["Age"] = st.number_input("Age: ", min_value=1, max_value=120, step=1, format="%d")

    form_values["Genetic Markers"] = st.selectbox("Genetic Markers :", ['Positive', 'Negative'])

    form_values["Autoantibodies"] = st.selectbox("Autoantibodies :", ['Positive', 'Negative'])

    form_values["Family History"] = st.selectbox("Family History :", ["Yes", "No"])

    form_values["Environmental Factors"] = st.selectbox("Environmental Factors :", ['Present', 'Absent'])

    form_values["Insulin Levels"] = st.number_input("Insulin Levels :", step=1, format="%d")

    form_values["BMI"] = st.number_input("BMI :", step=1, format="%d")

    form_values["Physical Activity"] = st.selectbox("Physical Activity :", ['High', 'Low', 'Moderate'])
    
    form_values["Dietary Habits"] = st.selectbox("Dietary Habits :", ['Healthy', 'Unhealthy'])
    
    form_values["Blood Pressure"] = st.number_input("Blood Pressure :", step=1, format="%d")
    
    form_values["Cholesterol Levels"] = st.number_input("Cholesterol Levels :", step=1, format="%d")
    
    form_values["Waist Circumference"] = st.number_input("Waist Circumference :", step=1, format="%d")
    
    form_values["Blood Glucose Levels"] = st.number_input("Blood Glucose Levels :", step=1, format="%d")
    
    form_values["Ethnicity"] = st.selectbox("Ethnicity :", ['Low Risk', 'High Risk'])
    
    form_values["Socioeconomic Factors"] = st.selectbox("Socioeconomic Factors :", ['Low', 'Medium', 'High'])
    
    form_values["Smoking Status"] = st.selectbox("Smoking Status :", ['Smoker', 'Non-Smoker'])
    
    form_values["Alcohol Consumption"] = st.selectbox("Alcohol Consumption :", ['High', 'Moderate', 'Low'])
    
    form_values["Glucose Tolerance Test"] = st.selectbox("Glucose Tolerance Test :", ['Normal', 'Abnormal'])
    
    form_values["History of PCOS"] = st.selectbox("History of PCOS :", ['No', 'Yes'])
    
    form_values["Previous Gestational Diabetes"] = st.selectbox("Previous Gestational Diabetes :", ['No', 'Yes'])
    
    form_values["Pregnancy History"] = st.selectbox("Pregnancy History :", ['Normal', 'Complications'])
    
    form_values["Weight Gain During Pregnancy"] = st.number_input("Weight Gain During Pregnancy :", step=1, format="%d")
    
    form_values["Pancreatic Health"] = st.number_input("Pancreatic Health :", step=1, format="%d")
    
    form_values["Pulmonary Function"] = st.number_input("Pulmonary Function :", step=1, format="%d")
    
    form_values["Cystic Fibrosis Diagnosis"] = st.selectbox("Cystic Fibrosis Diagnosis :", ['No', 'Yes'])
    
    form_values["Steroid Use History"] = st.selectbox("Steroid Use History :", ['No', 'Yes'])
    
    form_values["Genetic Testing"] = st.selectbox("Genetic Testing :", ['Positive', 'Negative'])
    
    form_values["Neurological Assessments"] = st.selectbox("Neurological Assessments :", [1, 2, 3])
    
    form_values["Liver Function Tests"] = st.selectbox("Liver Function Tests :", ['Normal', 'Abnormal'])
    
    form_values["Digestive Enzyme Levels"] = st.number_input("Digestive Enzyme Levels :", step=1, format="%d")
    
    form_values["Urine Test"] = st.selectbox("Urine Test :", ['Normal', 'Glucose Present', 'Protein Present', 'Ketones Present'])
    
    form_values["Birth Weight"] = st.number_input("Birth Weight(g) :", step=1, format="%d")
    
    form_values["Early Onset Symptoms"] = st.selectbox("Early Onset Symptoms :", ['No', 'Yes'])

    submit_button = st.form_submit_button(label= "Submit")

    if submit_button:
        if not all(form_values.values()):
            st.warning("Thông tin chưa đầy đủ!")
        else:
            data = pd.DataFrame(form_values,index=[name])
            st.write("Patient information :")
            st.dataframe(data.drop(["Target"], axis=1))

            user_data = data.astype(object)
            answer = prediction(user_data)
            st.write(f"Model prediction : {answer}")
