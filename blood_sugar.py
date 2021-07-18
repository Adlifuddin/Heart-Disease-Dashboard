import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import altair as alt

def blood_sugar(data):
    normal_blood_sugar = data.loc[data['fbs'] == 0]
    normal_blood_sugar_yes = normal_blood_sugar.loc[normal_blood_sugar['target'] == 1]
    normal_blood_sugar_no = len(normal_blood_sugar) - len(normal_blood_sugar_yes)

    high_blood_sugar = data.loc[data['fbs'] == 1]
    high_blood_sugar_yes = high_blood_sugar.loc[high_blood_sugar['target'] == 1]
    high_blood_sugar_no = len(high_blood_sugar) - len(high_blood_sugar_yes)

    fbs_data = pd.DataFrame({
        'Types': ['Normal (<=120)', 'High (>120)'],
        'b': [len(normal_blood_sugar_yes), len(high_blood_sugar_yes)],
        'c': [normal_blood_sugar_no, high_blood_sugar_no]
    })

    col1, col2 = st.beta_columns(2)
    # st.write('Data Length:', len(normal_blood_sugar+high_blood_sugar))

    with col1:
        bar = alt.Chart(fbs_data).mark_bar().encode(
            x=alt.X('Types', title='Blood Sugar Level'),
            y=alt.Y('b', title='Number of people'),
            color='Types'
        )
        st.subheader("Have Heart Disease based on Blood Sugar Level")
        st.altair_chart(bar, use_container_width=True)
        st.write('Normal:', len(normal_blood_sugar_yes))
        st.write('High:', len(high_blood_sugar_yes))

    with col2:
        bar = alt.Chart(fbs_data).mark_bar().encode(
            x=alt.X('Types', title='Blood Sugar Level'),
            y=alt.Y('c', title='Number of people'),
            color='Types'
        )
        st.subheader("Does not have Heart Disease based on Blood Sugar Level")
        st.altair_chart(bar, use_container_width=True)
        st.write('Normal:', normal_blood_sugar_no)
        st.write('High:', high_blood_sugar_no)