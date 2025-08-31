import streamlit as st

st.title("📧 Cold Email Generator")

url_input = st.text_input("Enter the URL :")
submit_button = st.button("Generate Email")
if submit_button:
   st.code("")