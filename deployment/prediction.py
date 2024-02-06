import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle
from PIL import Image


# Choice of input: Upload or Manual Input
inputType = st.selectbox("How would you like to input data ?", ["Upload Excel or CSV File", "Manual Input"])
st.markdown('---')



def run():

    # membuat title
    st.title("Bank Customer's Churn Prediction")

    # membuat subheader
    # st.subheader('Data Form Input Churn Classification')

    # menambahkan gambar 
    # image = Image.open('1.jpeg')
    # st.image(image)

    with st.form('Form Customer Churn Prediction'):

        # field customer id
        cust_id = st.number_input('Customer ID', min_value=0, max_value=10000)

        # field surname
        surname = st.text_input ('Name :')

        # field gender
        gender_options = ["Male", "Female"]
        gender = st.selectbox('Pilih Gender', gender_options)

        # field tenure
        age = st.number_input('Age', min_value=18, max_value=95)

        # field tenure
        tenure = st.number_input('Tenure', min_value=0, max_value=10, help='How long someone be customer (in years)')

    with st.form('Financial Information'):

        # field credit score
        creditscore = st.number_input('Credit Score', min_value=300 , max_value=900, help='How likely to pay a loan back on time, based on information from credit report')

        # field balance
        balance = st.number_input('Balance', min_value=0, max_value=350000, help='Amount of Balance')

        # field estimated salary
        estimated_salary = st.number_input('Estimated Salary', min_value=0, max_value=350000, help='Estimated customer salary')

    with st.form('Services and Membership'):

        # field num of products
        num_of_products = st.number_input('Num of Products', min_value=1 , max_value=4, help='Amount of product or services used (max=4)')

        # field has credit card
        HasCrCard_options = [0, 1]
        HasCrCard = st.selectbox('Has Credit Card', HasCrCard_options)
        st.write("0 = Don't have credit card")
        st.write("1 = Has credit card")
        
        # field active member
        is_active_member_options = [0,1]
        is_active_member = st.selectbox('Is Active Member', is_active_member_options)
        st.write("0 = Not active member")
        st.write("1 = Is active member")


        # submit button
        submitted = st.form_submit_button('Predict')


    # inference
    
    # with open('scaler.pkl', 'rb') as file_1:
    #     scaler = pickle.load(file_1)
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
    st.dataframe(data_inf)

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

