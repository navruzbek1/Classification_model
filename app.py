import numpy as np
import pandas as pd
import streamlit as st
import pickle
import time
from xgboost import XGBClassifier
st.title("Bemorlarni klassifikatsiyaga ajratuvchi model..")

number = st.number_input('Age')
option = st.selectbox(
     'Gender?',
     ('F', 'M'))
st.write('gender is selected:', option)
if option:
    if option == "M": 
        option = int(2)
        option = option/2
    if option == "F":
        option = int(1)
        
        
number1 = st.number_input('Height_cm')

number2 = st.number_input('Weight_kg')

number3 = st.number_input('Body_fat')

number4 = st.number_input('Diastolic')

number5 = st.number_input('Systolic')

number6 = st.number_input('GripForse')

number7 = st.number_input('sit_and_bend_forward_cm')
   
number8 = st.number_input('sit_ups_counts')

number9 = st.number_input('broad_jump_cm')

with open("xgb_model.pkl", 'rb') as file:
     model = pickle.load(file)
                   
result = st.button("Yuborish")
st.write(result)
if result:
        st.write()
        arr = np.array([number,option,number1,number2,number3,number4,number5,number6,number7,number8,number9])
        if arr.all():
        #arr = np.asanyarray([24,1,152,42,23,56,111,27,14,49,184])
        #arr
                with st.spinner('Wait for it...'):
                   time.sleep(1)
                st.success('Done!')
                pred = model.predict([arr])
                if pred==1:
                        st.success("Class: A")
                if  pred==2:
                        st.success("Class: B")
                if pred==3:
                        st.success("Class: C")
                if pred==4:
                        st.success("Class: D")
             
