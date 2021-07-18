import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import altair as alt

def vessel(data):
    vessel0 = data.loc[data['ca'] == 0]
    vessel0_yes = vessel0.loc[vessel0['target'] == 1]
    vessel0_no = len(vessel0) - len(vessel0_yes)

    vessel1 = data.loc[data['ca'] == 1]
    vessel1_yes = vessel1.loc[vessel1['target'] == 1]
    vessel1_no = len(vessel1) - len(vessel1_yes)

    vessel2 = data.loc[data['ca'] == 2]
    vessel2_yes = vessel2.loc[vessel2['target'] == 1]
    vessel2_no = len(vessel2) - len(vessel2_yes)

    vessel3 = data.loc[data['ca'] == 3]
    vessel3_yes = vessel3.loc[vessel3['target'] == 1]
    vessel3_no = len(vessel3) - len(vessel3_yes)

    vessel4 = data.loc[data['ca'] == 4]
    vessel4_yes = vessel4.loc[vessel4['target'] == 1]
    vessel4_no = len(vessel4) - len(vessel4_yes)

    ca_data = pd.DataFrame({
        'Types': ['0', '1', '2', '3', '4'],
        'b': [len(vessel0_yes), len(vessel1_yes), len(vessel2_yes), len(vessel3_yes), len(vessel4_yes)],
        'c': [vessel0_no, vessel1_no, vessel2_no, vessel3_no, vessel4_no]
    })

    col1, col2 = st.beta_columns(2)
    # st.write('Data Length:', len(vessel0+vessel1+vessel2+vessel3+vessel4))

    with col1:
        bar = alt.Chart(ca_data).mark_bar().encode(
            x=alt.X('Types', title='Num of major blood vessels'),
            y=alt.Y('b', title='Number of people'),
            color='Types'
        )
        st.subheader("Has Heart Disease based on Num of major blood vessels")
        st.altair_chart(bar, use_container_width=True)
        st.write('0:', len(vessel0_yes))
        st.write('1:', len(vessel1_yes))
        st.write('2:', len(vessel2_yes))
        st.write('3:', len(vessel3_yes))
        st.write('4:', len(vessel4_yes))

    with col2:
        bar = alt.Chart(ca_data).mark_bar().encode(
            x=alt.X('Types', title='Num of major blood vessels'),
            y=alt.Y('c', title='Number of people'),
            color='Types'
        )
        st.subheader("Does not has Heart Disease based on Num of major blood vessels")
        st.altair_chart(bar, use_container_width=True)
        st.write('0:', vessel0_no)
        st.write('1:', vessel1_no)
        st.write('2:', vessel2_no)
        st.write('3:', vessel3_no)
        st.write('4:', vessel4_no)