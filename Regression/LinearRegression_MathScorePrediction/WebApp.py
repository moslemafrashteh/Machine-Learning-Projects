import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
import sklearn as sk


# load the model from disk
filename = 'Regression/LinearRegression_MathScorePrediction/finalized_model.sav'
loaded_model = jb.load(filename)

# Creating the Titles and Header
st.title("Predicting the Grade of Math Course")
st.header("Predicting a Student's Math Score based on Her Characteristics.")

def load_data():
    df = pd.DataFrame({'Medu' : ['None' ,'Primary Education (4th grade)' ,'5th to 9th Grade' ,'Secondary Education','Higher Education']})
    return df
df = load_data()

age = st.slider("What is your age?", 6, 22)
absences = st.slider("Number of Absences", 0, 75)
freetime = st.slider("Number of Free Time", 1, 5)
MotherEducation = st.selectbox("What Is Your Mother Education?", df['Medu'])
MathGrade1 = st.slider("What is your Math Score 1?", 0, 20)
MathGrade2 = st.slider("What is your Math Score 2?", 0, 20)

# converting text input to numeric to get back predictions from backend model.
if MotherEducation == 'None':
    MotherEducation = 0
elif MotherEducation == 'Primary Education (4th grade)':
    MotherEducation = 1
elif MotherEducation == '5th to 9th Grade':
    MotherEducation = 2
elif MotherEducation == 'Secondary Education':
    MotherEducation = 3
else:
    MotherEducation = 4
    
# store the inputs
features = [age, absences, freetime, MotherEducation, MathGrade1, MathGrade2]
# convert user inputs into an array fr the model

int_features = [int(x) for x in features]
final_features = [np.array(int_features)]

# when the submit button is pressed
if st.button('Predict'):           
    prediction =  loaded_model.predict(final_features)
    st.success(f'Your Math Score would be: {round(prediction[0],2)}')
    st.balloons()
