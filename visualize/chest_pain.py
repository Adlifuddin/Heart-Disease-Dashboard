import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import altair as alt

def chest_pain(data):
    typical_angina = data.loc[data['cp'] == 0]
    typical_angina_yes = typical_angina.loc[typical_angina['target'] == 1]
    typical_angina_no = len(typical_angina) - len(typical_angina_yes)

    atypical_angina = data.loc[data['cp'] == 1]
    atypical_angina_yes = atypical_angina.loc[atypical_angina['target'] == 1]
    atypical_angina_no = len(atypical_angina) - len(atypical_angina_yes)

    non_anginal_pain = data.loc[data['cp'] == 2]
    non_anginal_pain_yes = non_anginal_pain.loc[non_anginal_pain['target'] == 1]
    non_anginal_pain_no = len(non_anginal_pain) - len(non_anginal_pain_yes)

    asymptomatic = data.loc[data['cp'] == 3]
    asymptomatic_yes = asymptomatic.loc[asymptomatic['target'] == 1]
    asymptomatic_no = len(asymptomatic) - len(asymptomatic_yes)

    cp_data = pd.DataFrame({
        'Types': ['Typical', 'Atypical', 'Non-anginal', 'Asymptomatic'],
        'b': [len(typical_angina_yes), len(atypical_angina_yes), len(non_anginal_pain_yes), len(asymptomatic_yes)],
        'c': [typical_angina_no, atypical_angina_no, non_anginal_pain_no, asymptomatic_no]
    })

    col1, col2 = st.beta_columns(2)
    # st.write('Data Length:', len(typical_angina+atypical_angina+non_anginal_pain+asymptomatic))

    with col1:
        bar = alt.Chart(cp_data).mark_bar().encode(
            x=alt.X('Types', title='Chest Pain Type'),
            y=alt.Y('b', title='Number of people'),
            color='Types'
        )
        st.subheader("Have Heart Disease based on Chest Pain Type")
        st.altair_chart(bar, use_container_width=True)
        st.write('Typical:', len(typical_angina_yes))
        st.write('Atypical:', len(atypical_angina_yes))
        st.write('Non-anginal:', len(non_anginal_pain_yes))
        st.write('Asymptomatic:', len(asymptomatic_yes))

    with col2:
        bar = alt.Chart(cp_data).mark_bar().encode(
            x=alt.X('Types', title='Chest Pain Type'),
            y=alt.Y('c', title='Number of people'),
            color='Types'
        )
        st.subheader("Does not have Heart Disease based on Chest Pain Type")
        st.altair_chart(bar, use_container_width=True)
        st.write('Typical:', typical_angina_no)
        st.write('Atypical:', atypical_angina_no)
        st.write('Non-anginal:', non_anginal_pain_no)
        st.write('Asymptomatic:', asymptomatic_no)