import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import altair as alt

def age(data):
    age1 = data.loc[(data['age'] >= 20) & (data['age'] < 40)]
    age1_yes = age1.loc[age1['target'] == 1]
    age1_no = len(age1) - len(age1_yes)

    age2 = data.loc[(data['age'] >= 40) & (data['age'] < 60)]
    age2_yes = age2.loc[age2['target'] == 1]
    age2_no = len(age2) - len(age2_yes)

    age3 = data.loc[(data['age'] >= 60) & (data['age'] < 80)]
    age3_yes = age3.loc[age3['target'] == 1]
    age3_no = len(age3) - len(age3_yes)

    age_data = pd.DataFrame({
        'Age': ['20-39', '40-59', '60-79'],
        'b': [len(age1_yes), len(age2_yes), len(age3_yes)],
        'c': [age1_no, age2_no, age3_no]
    })

    col1, col2 = st.beta_columns(2)
    # st.write('Data Length:', len(age1+age2+age3))

    with col1:
        bar = alt.Chart(age_data).mark_bar().encode(
            x=alt.X('Age', title='Age Groups'),
            y=alt.Y('b', title='Number of people'),
            color='Age'
        )
        st.subheader("Have Heart Disease based on Age Groups")
        st.altair_chart(bar, use_container_width=True)
        st.write('20-39:', len(age1_yes))
        st.write('40-59:', len(age2_yes))
        st.write('60-79:', len(age3_yes))

    with col2:
        bar = alt.Chart(age_data).mark_bar().encode(
            x=alt.X('Age', title='Age Groups'),
            y=alt.Y('c', title='Number of people'),
            color='Age'
        )
        st.subheader("Does not have Heart Disease based on Age Groups")
        st.altair_chart(bar, use_container_width=True)
        st.write('20-39:', age1_no)
        st.write('40-59:', age2_no)
        st.write('60-79:', age3_no)
