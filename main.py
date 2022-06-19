# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:53:45 2022

@author: Benk
"""
import streamlit as st
# import polars as pl
import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pickle
from sklearn.linear_model import LogisticRegression



# pip list --format=freeze
# streamlit run "C:\Users\Benk\Desktop\PowerBi\Personal Key Indicators of Heart Disease\main.py"


st.title('Heart Disease Prediction')

st.write("""
         ### Do you want to check your heart condition, this app built using machine learning method will help you to diagnose your heart condition!
         """)



# st.write(BMIdata)
# -------------------------------------------------------------------------

BMI=st.sidebar.selectbox("Select your BMI", ("Normal weight BMI  (18.5-25)", 
                             "Underweight BMI (< 18.5)" ,
                             "Overweight BMI (25-30)",
                             "Obese BMI (> 30)"))
Age=st.sidebar.selectbox("Select your age", 
                            ("18-24", 
                             "25-29" ,
                             "30-34",
                             "35-39",
                             "40-44",
                             "45-49",
                             "50-54",
                             "55-59",
                             "60-64",
                             "65-69",
                             "70-74",
                             "75-79",
                             "55-59",
                             "80 or older"))

Race=st.sidebar.selectbox("Select your Race", ("Asian", 
                             "Black" ,
                             "Hispanic",
                             "American Indian/Alaskan Native",
                             "White",
                             "Other"
                             ))

Gender=st.sidebar.selectbox("Select your gender", ("Female", 
                             "Male" ))

sleepTime = st.sidebar.number_input("How many hours on average do you sleep?", 0, 24, 7)

genHealth = st.sidebar.selectbox("How can you define your general health?",
                             options=("Good","Excellent", "Fair", "Very good", "Poor"))

physHealth = st.sidebar.number_input("Your physical health in the past 30 days was"
                                 , 0, 30, 0)
mentHealth = st.sidebar.number_input("Your mental health in the past 30 days"
                                 , 0, 30, 0)
physAct = st.sidebar.selectbox("Physical activity in the past month?"
                           , options=("No", "Yes"))

Smoking = st.sidebar.selectbox("Have you smoked at least 100 cigarettes in"
                          " your entire life (approx. 5 packs)?)",
                          options=("No", "Yes"))
alcoholDink = st.sidebar.selectbox("Do you have more than 14 drinks of alcohol (men)"
                                " or more than 7 (women) in a week?", options=("No", "Yes"))
stroke = st.sidebar.selectbox("Did you have a stroke?", options=("No", "Yes"))
diffWalk = st.sidebar.selectbox("Do you have serious difficulty walking"
                            " or climbing stairs?", options=("No", "Yes"))
diabetic = st.sidebar.selectbox("Have you ever had diabetes?",
                           options=("No", "Yes", "Yes, during pregnancy", "No, borderline diabetes"))
asthma = st.sidebar.selectbox("Do you have asthma?", options=("No", "Yes"))
kidneyDisease= st.sidebar.selectbox("Do you have kidney disease?", options=("No", "Yes"))
skinCancer = st.sidebar.selectbox("Do you have skin cancer?", options=("No", "Yes"))

dataToPredic = pd.DataFrame({
   "BMI": [BMI],
   "Smoking": [Smoking],
   "AlcoholDrinking": [alcoholDink],
   "Stroke": [stroke],
   "PhysicalHealth": [physHealth],
   "MentalHealth": [mentHealth],
   "DiffWalking": [diffWalk],
   "Gender": [Gender],
   "AgeCategory": [Age],
   "Race": [Race],
   "Diabetic": [diabetic],
   "PhysicalActivity": [physAct],
   "GenHealth": [genHealth],
   "SleepTime": [sleepTime],
   "Asthma": [asthma],
   "KidneyDisease": [kidneyDisease],
   "SkinCancer": [skinCancer]
 })

#-------------------------Mapping-------------------------------
dataToPredic.replace("Underweight BMI (< 18.5)",0,inplace=True)
dataToPredic.replace("Normal weight BMI  (18.5-25)",1,inplace=True)
dataToPredic.replace("Overweight BMI (25-30)",2,inplace=True)
dataToPredic.replace("Obese BMI (> 30)",3,inplace=True)

dataToPredic.replace("Yes",1,inplace=True)
dataToPredic.replace("No",0,inplace=True)
dataToPredic.replace("18-24",0,inplace=True)
dataToPredic.replace("25-29",1,inplace=True)
dataToPredic.replace("30-34",2,inplace=True)
dataToPredic.replace("35-39",3,inplace=True)
dataToPredic.replace("40-44",4,inplace=True)
dataToPredic.replace("45-49",5,inplace=True)
dataToPredic.replace("50-54",6,inplace=True)
dataToPredic.replace("55-59",7,inplace=True)
dataToPredic.replace("60-64",8,inplace=True)
dataToPredic.replace("65-69",9,inplace=True)
dataToPredic.replace("70-74",10,inplace=True)
dataToPredic.replace("75-79",11,inplace=True)
dataToPredic.replace("80 or older",13,inplace=True)


dataToPredic.replace("No, borderline diabetes",2,inplace=True)
dataToPredic.replace("Yes (during pregnancy)",3,inplace=True)


dataToPredic.replace("Excellent",0,inplace=True)
dataToPredic.replace("Good",1,inplace=True)
dataToPredic.replace("Fair",2,inplace=True)
dataToPredic.replace("Very good",3,inplace=True)
dataToPredic.replace("Poor",4,inplace=True)


dataToPredic.replace("White",0,inplace=True)
dataToPredic.replace("Other",1,inplace=True)
dataToPredic.replace("Black",2,inplace=True)
dataToPredic.replace("Hispanic",3,inplace=True)
dataToPredic.replace("Asian",4,inplace=True)
dataToPredic.replace("American Indian/Alaskan Native",4,inplace=True)


dataToPredic.replace("Female",0,inplace=True)
dataToPredic.replace("Male",1,inplace=True)


filename='finalized_model.sav'
loaded_model= pickle.load(open(filename, 'rb'))
Result=loaded_model.predict(dataToPredic)
ResultProb= loaded_model.predict_proba(dataToPredic)
ResultProb1=round(ResultProb[0][1] * 100, 2)




if st.button('Predict'):
 # st.write('your prediction:', Result, round(ResultProb[0][1] * 100, 2))
 if (ResultProb[0][1]>30):
  st.write('You are a person at risk for getting a heart disease, you have a', ResultProb[0][1], '% chance of getting a heart disease' )
 else:
  st.write('You are healthy, you have a', ResultProb[0][1], '% chance of getting a heart disease' )


 # array([[ chance of NOT having, chance of having 0.01151576]])
 # array([[0.98848424, 0.01151576]])


