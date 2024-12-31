import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle
import pandas as pd

## LOAD THE TRAINED MODEL

model=tf.keras.models.load_model('model.h5')

## LOAD THE PICKLE FILE
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('one_hot_enoder_geo.pkl','rb') as file:
    one_hot_enoder_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
  scaler=pickle.load(file)

##  STREAMLIT APP

st.title('Customer Churn Prediction')

## USER INPUT
geography=st.selectbox('Geography',one_hot_enoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
credit_score=st.number_input('Credit Score')
tenure=st.number_input('Tenure',0,10)
balance=st.number_input('Balance')
num_of_products=st.slider('Number of Products',1,4)
age=st.number_input('Age',min_value=18,max_value=92)
estimated_salary=st.number_input('Estimated Salary')
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

## ONE HOT ENCODED "GEOGRAPHY"

geo_encoded=one_hot_enoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=one_hot_enoder_geo.get_feature_names_out(['Geography']))
geo_encoded_df

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_scaler=scaler.transform(input_data)

prediction=model.predict(input_scaler)
prob=prediction[0][0]

if prob > 0.6:
  st.write('The customer is likely to churn')
else:
  st.write('The customer is likely to stay')
