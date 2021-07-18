import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, precision_score, f1_score, recall_score

# Logistic Regression
def pred_logisticRegression(train_input,train_target,test_input):
    # Train
    logisticRegression = LogisticRegression()
    model = logisticRegression.fit(train_input,train_target)
    pred(model,test_input)

# K-Nearest Neighbor
def pred_knn(train_input,train_target,test_input,n_neighbors=10):    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    model = knn.fit(train_input,train_target)
    pred(model,test_input)

# Decision Tree
def pred_decision_tree(train_input,train_target,test_input,ccp=0.0):
    decision_tree = DecisionTreeClassifier(criterion="gini", random_state=0, ccp_alpha=ccp)
    model = decision_tree.fit(train_input,train_target)
    pred(model,test_input)

# Random Forest
def pred_random_forest(train_input,train_target,test_input,estimator=100,ccp=0.0):
    rf = RandomForestClassifier(n_estimators=estimator)
    model = rf.fit(train_input,train_target)
    pred(model,test_input)

# multi:softprob/multi:softmax for multi class and binary:logistic for binary
def pred_xgboost(train_input,train_target,test_input,n_estimators=100,eta=0.1,gamma=0.3,min_child_weight=5,lr=0.1,max_depth=5,colsample_bytree=0.7):
    xgmodel = XGBClassifier(use_label_encoder=False,eta=eta,gamma=gamma, n_estimators=n_estimators, learning_rate=lr, min_child_weight=min_child_weight, max_depth=max_depth, colsample_bytree=colsample_bytree,objective="binary:logistic", eval_metric="mlogloss",verbosity=0)
    model = xgmodel.fit(train_input,train_target)
    pred(model,test_input)

# Standard Vector Machine
def pred_svm(train_input,train_target,test_input,gamma='auto'):
    svm_model = SVC(gamma=gamma)
    model = svm_model.fit(train_input,train_target)
    pred(model,test_input)

# Naive Bayes
def pred_naive_bayes(train_input,train_target,test_input):
    nb_model = GaussianNB()
    model = nb_model.fit(train_input,train_target)
    pred(model,test_input)

# Stochastic Gradient Descent
def pred_sgd(train_input,train_target,test_input,loss="hinge",penalty="l2"):
    nb_model = SGDClassifier(loss=loss,penalty=penalty)
    model = nb_model.fit(train_input,train_target)
    pred(model,test_input)

def pred(model,test_input):
    # Predict
    pred_data=model.predict(test_input)
    test_input['Predicted'] = pred_data
    if test_input['Predicted'][0] == 0:
        st.write('Result: Low Risk of Heart Disease')
    else:
        st.write('Result: High Risk of Heart Disease')