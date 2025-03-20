# app.py
import streamlit as st
from components.contact import render_contact_section

st.title("My Webpage")
# Other sections...
render_contact_section()