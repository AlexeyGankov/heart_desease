import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

ff = open("scaler.bin", "rb")
scaler = pickle.load(ff)
ff = open("model_1.bin", "rb")
model_1 = pickle.load(ff)

st.title('Heart diseases prediction\n On-line system')

st.sidebar.subheader('Please, enter information:')
    
display = ("male", "female")
gender = st.sidebar.selectbox("Your gender", list(range(len(display))), format_func=lambda x: display[x])
age = st.sidebar.number_input("Enter your age (years)", min_value=20, max_value=90,  step=None, format='%i')
age=age*365
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=220, value=150,  step=5, format='%i')
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=150, value=70,  step=1, format='%i')
ap_hi = st.sidebar.number_input("Systolic pressure mm Hg", min_value=60, max_value=300, value=120,  step=1, format='%i')
ap_lo = st.sidebar.number_input("Diastolic pressure mm Hg)", min_value=30, max_value=150, value=70,  step=1, format='%i')
imt = weight/(height**2)*10000
d_cholesterol = ('1','2','3')
cholesterol = st.sidebar.selectbox("Cholesterol indicator value?", (1,2,3))
cholesterol_2 = cholesterol_3 = 0
if cholesterol == 2:
	cholesterol_2 =1
if cholesterol == 3:
	cholesterol_3 =1
gluc = st.sidebar.selectbox("Glucose indicator value?", (1,2,3))
gluc_2 = gluc_3 = 0
if gluc == 2:
	gluc_2 =1
if gluc == 3:
	gluc_3 =1
d_smoke = ( "NO", "YES")
smoke = st.sidebar.selectbox("Are you a tobacco smoker?", list(range(len(d_smoke))), format_func=lambda x: d_smoke[x])
d_active = ( "NO", "YES")
active =  st.sidebar.selectbox("Do you lead an active lifestyle?", list(range(len(d_active))), format_func=lambda x: d_active[x])


input_test = pd.DataFrame([ [age,height, weight,ap_hi, ap_lo,imt,gender, smoke, active, cholesterol_2,cholesterol_3,gluc_3]] ,columns = ['age','height', 'weight','ap_hi','ap_lo','imt','gender', 'smoke', 'active', 'cholesterol_2','cholesterol_3','gluc_3'])
st.subheader('Your request data:')
st.write(input_test)
scale_columns = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'imt']
test_s= scaler.transform(input_test[scale_columns])
test_s = pd.DataFrame(test_s)
test_s.columns = scale_columns
input_test.drop(scale_columns, axis=1, inplace = True)
testf = pd.concat([input_test, test_s], axis=1)
columnsTitles = ['age','height', 'weight','ap_hi','ap_lo','imt','gender', 'smoke', 'active', 'cholesterol_2','cholesterol_3','gluc_3']
testf2 = testf.reindex(columns=columnsTitles)
predict_p = model_1.predict_proba(testf2)
res_p = predict_p[:,1][0]
st.subheader('Result:')
st.write('Heart disease predict indicator value', res_p)
if res_p > 0.75:
	st.write("You have high risk of heart diseases!")
if (res_p >0.5) and (res_p < 0.75):
	st.write("You have middle risk of heart diseases")
if (res_p <0.5) :
	st.write("You have low risk of heart diseases")
st.subheader('More advices:')
more_d = ' '
if ap_hi > 180 or ap_lo > 110:
	more_d = more_d + "Your blood pressure is high -  control it and you need to visit a doctor <br>   "
if imt > 25:
	more_d = more_d + "You have extra weight. Control it! <br>  "
if imt < 18:
	more_d = more_d + "You are underweight - visit an endocrinologist <br> "
if smoke == 1:
	more_d = more_d + "Don't smoke! <br>  "
if active == 0:
	more_d = more_d + " Go in for sports! <br>  "
if len(more_d) >1:
	st.write(more_d, unsafe_allow_html=True)
else:
	st.write("No more advices", unsafe_allow_html=True)


