import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle
import json
from PIL import Image

def run():

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
    # st.write("""
    #         <h1 style='text-align: center;'Bank Customer Churn Prediction</h1>
    #     """, unsafe_allow_html=True)


    tab1, tab2, tab3 = st.tabs(["Prediction Form", "Bulk Predictions", "EDA"])
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
            # # Create a new column "NewTenure"
            # data_inf_drop["NewTenure"] = data_inf_drop["tenure"] / data_inf_drop["age"]

        # predict
            y_pred_inf = model.predict(data_inf_drop)
            if y_pred_inf == 0:
                st.write('### Not Churn')
            else :
                st.write('### Churn')

        with tab2:

            st.write('## Bank Customer Churn Prediction')
            st.write('### Customer Data')

            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

            #Condition upload data
            if uploaded_file is not None:
                df_uploaded = pd.read_csv(uploaded_file)

                # Cek dan imputasi nilai NaN jika diperlukan
                if df_uploaded.isnull().values.any():
                    # Misalnya, isi nilai NaN dengan nilai median
                    df_uploaded = df_uploaded.fillna(df_uploaded.median())
                # Condition jika ada kolom yang perlu di drop
                if 'CustomerId' in df_uploaded.columns:
                    #Drop kolom
                    df_uploaded.drop('CustomerId', axis=1, inplace=True)

                if 'Surname' in df_uploaded.columns:
                    df_uploaded.drop('Surname', axis=1, inplace=True)

                if 'tenure' in df_uploaded.columns:
                    # df_uploaded['NewTenure']= df_uploaded["tenure"] / df_uploaded["age"]

                    df_uploaded.columns = ['Surname', 'geography', 'age', 'gender', 'tenure', 'credit_score', 'balance', 'estimated_salary', 'num_of_products', 'has_credit_card', 'active_member']

                y_pred_inf = model.predict(df_uploaded)
                #Menambahkan hasil predict ke dataframe
                df_uploaded['churn_prediction'] = np.array[y_pred_inf]
                
                st.write('### Dataframe Bulk Prediction')
                st.markdown('--------')
                st.dataframe(df_uploaded)

    with tab3:
        # Menambahkan hyperlink ke dashboard Looker
        looker_dashboard_url = "https://lookerstudio.google.com/s/omKUXbpaEps"
        st.markdown(f"Go to Looker Dashboard: [{looker_dashboard_url}]({looker_dashboard_url})")

if __name__ == '__main__':
    run()

