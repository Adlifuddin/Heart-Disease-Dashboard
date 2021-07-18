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
def logisticRegression(train_input,train_target,test_input,test_target):
    # Train
    logisticRegression = LogisticRegression()
    model = logisticRegression.fit(train_input,train_target)
    pred(model,test_input,test_target)

# K-Nearest Neighbor
def knn(train_input,train_target,test_input,test_target,n_neighbors=10):    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    model = knn.fit(train_input,train_target)
    pred(model,test_input,test_target)

# Decision Tree
def decision_tree(train_input,train_target,test_input,test_target,ccp=0.0):
    decision_tree = DecisionTreeClassifier(criterion="gini", random_state=0, ccp_alpha=ccp)
    model = decision_tree.fit(train_input,train_target)
    pred(model,test_input,test_target)

# Random Forest
def random_forest(train_input,train_target,test_input,test_target,estimator=100,ccp=0.0):
    rf = RandomForestClassifier(n_estimators=estimator)
    model = rf.fit(train_input,train_target)
    pred(model,test_input,test_target)

# multi:softprob/multi:softmax for multi class and binary:logistic for binary
def xgboost(train_input,train_target,test_input,test_target,n_estimators=100,eta=0.1,gamma=0.3,min_child_weight=5,lr=0.1,max_depth=5,colsample_bytree=0.7):
    xgmodel = XGBClassifier(use_label_encoder=False,eta=eta,gamma=gamma, n_estimators=n_estimators, learning_rate=lr, min_child_weight=min_child_weight, max_depth=max_depth, colsample_bytree=colsample_bytree,objective="binary:logistic", eval_metric="mlogloss",verbosity=0)
    model = xgmodel.fit(train_input,train_target)
    pred(model,test_input,test_target)

# Standard Vector Machine
def svm(train_input,train_target,test_input,test_target,gamma='auto'):
    svm_model = SVC(gamma=gamma)
    model = svm_model.fit(train_input,train_target)
    pred(model,test_input,test_target)

# Naive Bayes
def naive_bayes(train_input,train_target,test_input,test_target):
    nb_model = GaussianNB()
    model = nb_model.fit(train_input,train_target)
    pred(model,test_input,test_target)

# Stochastic Gradient Descent
def sgd(train_input,train_target,test_input,test_target,loss="hinge",penalty="l2"):
    nb_model = SGDClassifier(loss=loss,penalty=penalty)
    model = nb_model.fit(train_input,train_target)
    pred(model,test_input,test_target)

def pred(model,test_input,test_target):
    # Predict
    pred_data=model.predict(test_input)
    conf_matrix = confusion_matrix(test_target,pred_data)
    acc_score = accuracy_score(test_target, pred_data)
    pre_score = precision_score(test_target, pred_data, average="macro")
    re_score = recall_score(test_target, pred_data, average="macro")
    f_score = f1_score(test_target, pred_data, average="macro")

    st.write(f"Accuracy: {str(round(acc_score*100,2))} %")
    st.write(f"Precision: {str(round(pre_score*100,2))} %")
    st.write(f"Recall: {str(round(re_score*100,2))} %")
    st.write(f"F1-Score: {str(round(f_score*100,2))} %")
    st.write(conf_matrix)