import streamlit as st
import pathlib


st.html("<h1 style='text-align: center; font-size: 3rem;'>Medical Jarvis</h1>")

with st.form(key="upload_form"):
    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    submit = st.form_submit_button("Submit")
    
if submit:
    st.success("Submitted Image Sucessfully")
    
    if uploaded_image is not None:
         st.image(uploaded_image, caption="Uploaded Image", width=250)
    else:
         st.warning("No image uploaded.")