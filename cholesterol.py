import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import altair as alt

def cholesterol(data):
    normal_cholesterol = data.loc[data['chol'] < 200]
    normal_cholesterol_yes = normal_cholesterol.loc[normal_cholesterol['target'] == 1]
    normal_cholesterol_no = len(normal_cholesterol) - len(normal_cholesterol_yes)

    high_cholesterol = data.loc[data['chol'] >= 200]
    high_cholesterol_yes = high_cholesterol.loc[high_cholesterol['target'] == 1]
    high_cholesterol_no = len(high_cholesterol) - len(high_cholesterol_yes)

    chol_data = pd.DataFrame({
        'Types': ['Normal (<200)', 'High (>=200)'],
        'b': [len(normal_cholesterol_yes), len(high_cholesterol_yes)],
        'c': [normal_cholesterol_no, high_cholesterol_no]
    })

    col1, col2 = st.beta_columns(2)
    # st.write('Data Length:', len(normal_cholesterol+high_cholesterol))

    with col1:
        bar = alt.Chart(chol_data).mark_bar().encode(
            x=alt.X('Types', title='Cholesterol Level'),
            y=alt.Y('b', title='Number of people'),
            color='Types'
        )
        st.subheader("Have Heart Disease based on Cholesterol Level")
        st.altair_chart(bar, use_container_width=True)
        st.write('Normal:', len(normal_cholesterol_yes))
        st.write('High:', len(high_cholesterol_yes))

    with col2:
        bar = alt.Chart(chol_data).mark_bar().encode(
            x=alt.X('Types', title='Cholesterol Level'),
            y=alt.Y('c', title='Number of people'),
            color='Types'
        )
        st.subheader("Does not have Heart Disease based on Cholesterol Level")
        st.altair_chart(bar, use_container_width=True)
        st.write('Normal:', normal_cholesterol_no)
        st.write('High:', high_cholesterol_no)