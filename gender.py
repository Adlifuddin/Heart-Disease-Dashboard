import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import altair as alt

def gender(data):
    male = data.loc[data['sex'] == 1]
    male_yes = male.loc[male['target'] == 1]
    male_no = len(male) - len(male_yes)

    female = data.loc[data['sex'] == 0]
    female_yes = female.loc[female['target'] == 1]
    female_no = len(female) - len(female_yes)

    gender_data = pd.DataFrame({
        'Gender': ['Male', 'Female'],
        'b': [len(male_yes), len(female_yes)],
        'c': [male_no, female_no]
    })

    col1, col2 = st.beta_columns(2)
    # st.write('Data Length:', len(male+female))

    with col1:
        bar = alt.Chart(gender_data).mark_bar().encode(
            x=alt.X('Gender', title='Gender'),
            y=alt.Y('b', title='Number of people'),
            color='Gender'
        )
        st.subheader("Have Heart Disease based on Gender")
        st.altair_chart(bar, use_container_width=True)
        st.write('Male:', len(male_yes))
        st.write('Female:', len(female_yes))

    with col2:
        bar = alt.Chart(gender_data).mark_bar().encode(
            x=alt.X('Gender', title='Gender'),
            y=alt.Y('c', title='Number of people'),
            color='Gender'
        )
        st.subheader("Does not have Heart Disease based on Gender")
        st.altair_chart(bar, use_container_width=True)
        st.write('Male:', male_no)
        st.write('Female:', female_no)
