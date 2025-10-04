import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")
st.set_page_config(layout="wide")


# st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.slider("sepal_length", 4.3, 7.9, 5.4)
    sepal_width = st.slider("sepal_width", 2.0, 4.4, 3.4)
    petal_length = st.slider("petal_length", 1.0, 6.9, 1.3)
    petal_width = st.slider("petal_width", 0.1, 2.5, 0.2)

    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features
col1, col2 = st.columns(2,border=True)
with col1: 
    with st.container(border=True):
        st.markdown("## User Input Parameters")
        df = user_input_features()
    with st.container(border=True):
        df_show = df.T.reset_index().rename(columns={"index": "Dimensions", 0:"Value"})
        st.dataframe(df_show, hide_index=True)

iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X=X, y=y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

target_map = {}
for i, target in enumerate(iris.target_names):
    target_map[i] = target.capitalize()
# st.subheader("Class Labels and their corresponding index number")
# st.dataframe(iris.target_names, hide_index=False)

prediciton_proba_data = (
    pd.DataFrame(prediction_proba.T)
    .reset_index()
    .rename(columns={"index": "Target", 0: "Prediction Probability"})
    )
prediciton_proba_data["Flower_Type"] = prediciton_proba_data["Target"].replace(target_map)
prediciton_proba_data = prediciton_proba_data[["Flower_Type", "Prediction Probability"]]
proba_bar_chart = px.bar(prediciton_proba_data, x='Flower_Type', y='Prediction Probability')

with col2:
    with st.container(border=True):
        st.markdown("## Prediction")
        st.markdown(f"##### _Flower Type Predicted_: __{iris.target_names[prediction][0].capitalize()}__")
    with st.container(border=True):
        st.markdown("##### _Prediction Probability Bar Chart_")
        st.plotly_chart(proba_bar_chart, key="prediction_proba_chart")
        with st.expander("_Prediction Probability Table_"):
            st.markdown("##### _Prediction Probability Table_")
            st.dataframe(prediciton_proba_data, hide_index=True, width='stretch')
