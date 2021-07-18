import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import altair as alt

def blood_pressure(data):
    normal_blood_pressure = data.loc[data['trestbps'] < 140]
    normal_blood_pressure_yes = normal_blood_pressure.loc[normal_blood_pressure['target'] == 1]
    normal_blood_pressure_no = len(normal_blood_pressure) - len(normal_blood_pressure_yes)

    high_blood_pressure = data.loc[data['trestbps'] >= 140]
    high_blood_pressure_yes = high_blood_pressure.loc[high_blood_pressure['target'] == 1]
    high_blood_pressure_no = len(high_blood_pressure) - len(high_blood_pressure_yes)

    trestbps_data = pd.DataFrame({
        'Types': ['Normal (<140)', 'High (>=140)'],
        'b': [len(normal_blood_pressure_yes), len(high_blood_pressure_yes)],
        'c': [normal_blood_pressure_no, high_blood_pressure_no]
    })

    col1, col2 = st.beta_columns(2)
    # st.write('Data Length:', len(normal_blood_pressure+high_blood_pressure))

    with col1:
        bar = alt.Chart(trestbps_data).mark_bar().encode(
            x=alt.X('Types', title='Blood Pressure Level'),
            y=alt.Y('b', title='Number of people'),
            color='Types'
        )
        st.subheader("Have Heart Disease based on Blood Pressure Level")
        st.altair_chart(bar, use_container_width=True)
        st.write('Normal:', len(normal_blood_pressure_yes))
        st.write('High:', len(high_blood_pressure_yes))

    with col2:
        bar = alt.Chart(trestbps_data).mark_bar().encode(
            x=alt.X('Types', title='Blood Pressure Level'),
            y=alt.Y('c', title='Number of people'),
            color='Types'
        )
        st.subheader("Does not have Heart Disease based on Blood Pressure Level")
        st.altair_chart(bar, use_container_width=True)
        st.write('Normal:', normal_blood_pressure_no)
        st.write('High:', high_blood_pressure_no)