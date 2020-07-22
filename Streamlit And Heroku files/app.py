import streamlit as st
import joblib
import numpy as np


model=joblib.load('model.pkl')
count_vect=joblib.load('count_vect.pkl')

def classifier(email_text):
    input=[email_text]
    classified=model.predict(count_vect.transform(input))
    final=int(classified)
    return final

def main():
    html_temp='''
    <div style="background-color:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">Email Spam Detector </h2>
        </div>
         <br>
         <br>

    '''


    st.markdown(html_temp,unsafe_allow_html=True)

    email_text=st.text_input("Email Text","Type or paste the text of the email here")

    if st.button("Predict"):
        output= classifier(email_text)
        if output==0:
            output2="Not a Spam"
        else:
            output2="a Spam"
        st.success(f'This email is {output2}')

if __name__=='__main__':
    main()








