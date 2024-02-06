import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle
from sklearn.metrics._dist_metrics import EuclideanDistance
from sklearn.metrics.pairwise import euclidean_distances

from PIL import Image

def run():

    # membuat title
    st.title("Bank Customer's Churn Prediction")

    # Create Form
    # with st.form(key='Form Parameters'):
    surname = st.text_input('Name', value='', help='Customer Name')
    cust_id = st.number_input('Customer ID', min_value=0, max_value=10000)

    col_left, col_mid, col_right = st.columns([3, 2, 2])

    gender =  col_left.selectbox('Gender', ('Male', 'Female'), index=0)
    with col_mid:
        age = st.number_input('Age', min_value=18, max_value=95) 
    with col_right:
        tenure = st.number_input('Tenure (Year)', min_value=1, max_value=10, step=1)

    st.markdown('-----------------')
    st.write('Financial Information')

    col_left, col_mid, col_right = st.columns([3, 2, 2])
    with col_left:
        creditscore = st.number_input('Credit Score', min_value=300 , max_value=900, help='How likely to pay a loan back on time, based on information from credit report')
    with col_mid:
        balance = st.number_input('Balance', min_value=0, max_value=350000, help='Amount of Balance')
    with col_right:
        estimated_salary = st.number_input('Estimated Salary', min_value=0, max_value=350000, help='Estimated customer salary')

    st.markdown('-----------------')
    st.write('Services and Membership')

    num_of_products = st.number_input('Num of Products', min_value=1 , max_value=4, help='Amount of product or services used (max=4)')

    col1, col2 = st.columns([1, 1])
    HasCrCard = col1.radio(label='Has Credit Card?', options=[0, 1], help='Choose 1 for have credit card')
    is_active_member = col2.radio(label='Is Active Member?', options=[0,1], help='Choose 1 for active member')


    # submit button
    submitted = st.button('Predict')
    # submitted = st.form_submit_button('Predict')

    with open('model_knn.pkl', 'rb') as file_2:
        model = pickle.load(file_2)

    
    data_inf = {
        'CustomerId' : cust_id,
        'Surname' : surname,
        'age' : age,
        'gender' : gender,
        'tenure' : tenure,
        'credit_score' : creditscore,
        'balance' : balance,
        'estimated_salary' : estimated_salary,
        'num_of_products' : num_of_products,
        'HasCrCard' : HasCrCard,
        'is_active_member' : is_active_member
    }

    # memasukkan data inference ke dataframe
    data_inf = pd.DataFrame([data_inf])
    # st.dataframe(data_inf)

    # logic ketika predict button ditekan
    if submitted:
        data_inf_drop = data_inf.drop(['CustomerId', 'Surname'], axis=1)
        # Create a new column "NewTenure"
        data_feature_final = data_inf_drop["NewTenure"] = data_inf_drop["tenure"] / data_inf_drop["age"]

        # data_inf_scaled = scaler.transform(data_inf_drop)

    # predict
        y_pred_inf = model.predict(data_feature_final)
        st.write('## Customer :', str(int(y_pred_inf)))
        st.write('### Not Churn : 0, Churn : 1')
        st.write('### Churn : 1')

if __name__ == '__main__':
    run()