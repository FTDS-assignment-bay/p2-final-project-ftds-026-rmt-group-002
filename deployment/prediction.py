import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle
import json
from PIL import Image

def run():
    # membagi layout menjadi 3 agar dapat diletakkan di tengah
    col_left, col_mid, col_right = st.columns([2, 2, 2])
    # menambahkan gambar
    image = Image.open('image.webp')
    with col_mid :
        st.image(image)
        # membuat title
        st.title("ChurnSight")

    col_left, col_mid, col_right = st.columns([1.5, 3, 1.5])
    with col_mid :
        st.write("#### Bank Customer Churn Prediction")


    tab1, tab2 = st.tabs(["Prediction Form", "EDA"])
    #tab1 form input prediction
    with tab1:
        # Create Form
        st.markdown('-----------------')
        st.write('Identity Information')

        surname = st.text_input('Name', value='', help='Customer Name')
        cust_id = st.number_input('Customer ID', min_value=0, max_value=10000)
        geography = st.selectbox('Country', ('France', 'Spain', 'Germany'))

        col_left, col_mid, col_right = st.columns([2, 2, 2])

        gender =  col_left.selectbox('gender', ('Male', 'Female'), index=0)
        with col_mid:
            age = st.number_input('Age', min_value=18, max_value=95) 
        with col_right:
            tenure = st.number_input('Tenure (Year)', min_value=1, max_value=10, step=1)

        st.markdown('-----------------')
        st.write('Financial Information')

        col_left, col_mid, col_right = st.columns([2, 2, 2])
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
        has_credit_card = col1.radio(label='Has Credit Card?', options=['no', 'yes'])
        active_member = col2.radio(label='Is Active Member?', options=['no', 'yes'])


        # submit button
        submitted = st.button('Predict')

        # load files
        with open('model_xgb.pkl', 'rb') as file_2:
            model = pickle.load(file_2)
        with open('list_cat_cols.txt', 'rb') as file_3:
            cat_cols = json.load(file_3)
        with open('list_num_cols.txt', 'rb') as file_4:
            num_cols = json.load(file_4)

        
        data_inf = {
            'CustomerId' : cust_id,
            'Surname' : surname,
            'geography' : geography,
            'age' : age,
            'gender' : gender,
            'tenure' : tenure,
            'credit_score' : creditscore,
            'balance' : balance,
            'estimated_salary' : estimated_salary,
            'num_of_products' : num_of_products,
            'has_credit_card' : has_credit_card,
            'active_member' : active_member
        }

        # memasukkan data inference ke dataframe
        data_inf = pd.DataFrame([data_inf])
        st.dataframe(data_inf)

        # logic ketika predict button ditekan
        if submitted:
            data_inf_drop = data_inf.drop(['CustomerId', 'Surname'], axis=1)

        # predict
            y_pred_inf = model.predict(data_inf_drop)

            # conditional if 
            if y_pred_inf == 0:
                st.write('### Not Churn')
            else :
                st.write('### Churn')
                st.markdown('-----------------')
                # menampilkan rekomendasi jika terprediksi churn
                st.write('#### Based on analysis of customer churn data patterns, we make these following recommendations: ')
                st.markdown("""
- Need to make offers and marketing more attractive to female customers.
- Evaluate factors that make Spain have lower churn, and apply the same strategy to other countries.
- Evaluate products  according to the needs of customers who have high incomes.
- Create a program that can increase the profits of customers who actively make transactions and have high balances, such as giving vouchers/other benefits for every transaction with a certain value.
- Create "bundling" or packages containing several products, where the bundling also provides special benefits that cannot be obtained if only use certain products/without bundling.
""")
        # tab 2 eda
        with tab2:
            looker_dashboard_url = "https://lookerstudio.google.com/s/omKUXbpaEps"
            st.markdown(f"Go to Looker Dashboard: [{looker_dashboard_url}]({looker_dashboard_url})")

if __name__ == '__main__':
    run()

