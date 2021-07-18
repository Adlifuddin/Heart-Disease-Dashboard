import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import altair as alt

def overall(data):
    has_hd = data.loc[data['target'] == 1]
    no_hd = len(data) - len(has_hd)

    overall_data = pd.DataFrame({
        'overall': ['Have Heart Disease', 'Do not have Heart Disease'],
        'b': [len(has_hd), no_hd],
    })

    bar = alt.Chart(overall_data).mark_bar().encode(
        x=alt.X('overall', title='Heart Disease'),
        y=alt.Y('b', title='Number of people'),
        color='overall'
    )
    st.subheader("Overall Analysis of Heart Disease")
    st.altair_chart(bar, use_container_width=True)
    st.write('Have Heart Disease:', len(has_hd))
    st.write('Do not have Heart Disease:', no_hd)