import streamlit as st

st.set_page_config(
    page_title="Homework Manager",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define each HW page
hw1 = st.Page("HW/hw-1.py", title="HW 1")
hw2 = st.Page("HW/hw-2.py", title="HW 2")
hw3 = st.Page("HW/hw-3.py", title="HW 3", default=True)

# Build navigation
pg = st.navigation([hw1, hw2, hw3])

# Run selected page
pg.run()