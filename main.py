import xgboost
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import altair as alt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from gender import gender
from chest_pain import chest_pain
from blood_sugar import blood_sugar
from blood_pressure import blood_pressure
from cholesterol import cholesterol
from vessel import vessel
from age import age
from model import *
from pred_model import *

data = pd.read_csv('heart.csv')

st.set_page_config(layout="wide")
st.title("Heart Disease Analysis and Prediction")
st.header("Data")
st.write(data)
st.write('Data Length:', len(data))

# st.header("Preprocessing Data")
# Drop Missing Values
data = data.dropna(axis=0, how ='any') 

st.header("Data Visualization")

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


gender(data)
chest_pain(data)
blood_sugar(data)
blood_pressure(data)
cholesterol(data)
vessel(data)
age(data)

st.header("Classification Models used for prediction")

# Handle Outlier
num_col = list(data.columns[data.dtypes != 'object'])
# remove target
num_col.remove('target')
# Use zscore
for col in num_col:
    data = data[(np.abs(zscore(data[col])) < 3)]

# Handle Imbalance
over_sampling = RandomOverSampler(random_state=42)
x_res, y_res = over_sampling.fit_resample(data.loc[:, data.columns != 'target'], data['target'])
data = pd.concat([x_res,y_res], axis=1)

# Handle Normalization
scaler = StandardScaler()
data_scaler = scaler.fit(data[num_col])
df_scaled = data_scaler.transform(data[num_col])
df_scaled = pd.DataFrame(df_scaled,columns=num_col) 
data=data.drop(num_col,axis=1)
data=pd.concat([data,df_scaled],axis=1, join='inner')

input = data.drop(['target'],axis=1)
target = data['target']
X_train, X_test, y_train, y_test = train_test_split(input,target,test_size=0.2)

col1, col2, col3, col4 = st.beta_columns(4)
col5, col6, col7, col8 = st.beta_columns(4)

with col1:
    st.subheader("Logistic Regression")
    logisticRegression(X_train, y_train, X_test, y_test)

with col2:
    st.subheader("K-Nearest Neighbor")
    knn(X_train, y_train, X_test, y_test)

with col3:
    st.subheader("Decision Tree")
    decision_tree(X_train, y_train, X_test, y_test)

with col4:
    st.subheader("Random Forest")
    random_forest(X_train, y_train, X_test, y_test)

with col5:
    st.subheader("Support Vector Classification")
    svm(X_train, y_train, X_test, y_test)

with col6:
    st.subheader("Xgboost")
    xgboost(X_train, y_train, X_test, y_test)

with col7:
    st.subheader("Gaussian Naive Bayes")
    naive_bayes(X_train, y_train, X_test, y_test)

with col8:
    st.subheader("Stochastic Gradient Descent")
    sgd(X_train, y_train, X_test, y_test)

st.header("Heart Disease Risk Prediction")

show = False

with st.form("risk_form"):

    age_input = st.number_input('Age? (22 - 100)',22,100)

    gender_input = st.radio("Gender? ",('Male', 'Female'))
    if gender_input == 'Male':
        gender_input = 1
    else:
        gender_input = 0

    cpt_input = st.radio("Chest Pain Type? ",('Typical', 'Atypical', 'Non-anginal', 'Asymptomatic'))
    if cpt_input == 'Typical':
        cpt_input = 0
    elif cpt_input == 'Atypical':
        cpt_input = 1
    elif cpt_input == 'Non-anginal':
        cpt_input = 2 
    else:
        cpt_input = 3

    diabetes_input = st.radio("Diabetes or (fasting blood sugar > 120 mg/dl) ?",('Yes', 'No'))
    if diabetes_input == 'Yes':
        diabetes_input = 1
    else:
        diabetes_input = 0

    chol_input = st.radio("Dyslipidemia or (cholesterol >= 200 mg/dl) ? ",('Yes', 'No'))
    if chol_input == 'Yes':
        chol_input = 1
    else:
        chol_input = 0

    bp_input = st.radio("Hypertension or (blood pressure >- 140 mm Hg) ? ",('Yes', 'No'))
    if bp_input == 'Yes':
        bp_input = 1
    else:
        bp_input = 0

    model_option = st.selectbox(
        'Machine Learning Model?',
        ('Logistic Regression', 'K-Nearest Neighbor', 'Decision Tree', 'Random Forest', 'Support Vector Classification',
        'Xgboost', 'Gaussian Naive Bayes', 'Stochastic Gradient Descent'))

    # Submit button
    submitted = st.form_submit_button("Submit")
    if submitted:
        test = {'age': [age_input], 'sex': [gender_input], 'cp': [cpt_input], 'fbs': [diabetes_input],
        'chol': [chol_input], 'trestbps': [bp_input]}
        show = True
        check_data = pd.DataFrame(test)

if show:
    pred_data = pd.read_csv('heart_risk_pred.csv')

    # Preprocessing
    # Drop Missing Values
    pred_data = pred_data.dropna(axis=0, how ='any')

    pred_data.loc[pred_data['chol'] < 200, 'chol'] = 0
    pred_data.loc[pred_data['chol'] >= 200, 'chol'] = 1

    pred_data.loc[pred_data['trestbps'] < 140, 'trestbps'] = 0
    pred_data.loc[pred_data['trestbps'] >= 140, 'trestbps'] = 1

    # Handle Outlier
    num_col = list(pred_data.columns[pred_data.dtypes != 'object'])
    # remove target
    num_col.remove('target')
    # Use zscore
    for col in num_col:
        pred_data = pred_data[(np.abs(zscore(pred_data[col])) < 3)]
    
    # Handle Imbalance
    over_sampling = RandomOverSampler(random_state=42)
    x_res, y_res = over_sampling.fit_resample(pred_data.loc[:, pred_data.columns != 'target'], pred_data['target'])
    pred_data = pd.concat([x_res,y_res], axis=1)

    # Handle Normalization
    scaler = StandardScaler()
    data_scaler = scaler.fit(pred_data[num_col])
    df_scaled = data_scaler.transform(pred_data[num_col])
    df_scaled = pd.DataFrame(df_scaled,columns=num_col) 
    pred_data=pred_data.drop(num_col,axis=1)
    pred_data=pd.concat([pred_data,df_scaled],axis=1, join='inner')

    # st.write(pred_data)

    input_pred = pred_data.drop(['target'],axis=1)
    target_pred = pred_data['target']
    X_train_pred, X_test_pred, y_train_pred, y_test_pred = train_test_split(input_pred,target_pred,test_size=0.2)

    st.write(f'Model: {model_option}')
    if model_option == 'Logistic Regression':
        pred_logisticRegression(X_train_pred, y_train_pred, check_data)

    elif model_option == 'K-Nearest Neighbor':
        pred_knn(X_train_pred, y_train_pred, check_data)

    elif model_option == 'Decision Tree':
        pred_decision_tree(X_train_pred, y_train_pred, check_data)

    elif model_option == 'Random Forest':
        pred_random_forest(X_train_pred, y_train_pred, check_data)

    elif model_option == 'Support Vector Classification':
        pred_svm(X_train_pred, y_train_pred, check_data)

    elif model_option == 'Xgboost':
        pred_xgboost(X_train_pred, y_train_pred, check_data)

    elif model_option == 'Gaussian Naive Bayes':
        pred_naive_bayes(X_train_pred, y_train_pred, check_data)

    elif model_option == 'Stochastic Gradient Descent':
        pred_sgd(X_train_pred, y_train_pred, check_data)
