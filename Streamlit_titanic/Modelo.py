import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



@st.cache_data
def modelo(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(criterion = 'entropy', 
                                   max_depth =  5, n_estimators = 100, random_state = 42)
    return model.fit(X_train, y_train)

def convert_pclas(pclas): 
    if pclas == 'Baja':
        return 3
    elif pclas == 'Media':
        return 2
    else:
        return 1
def convert_s(sex):
    if sex == 'Hombre':
        return 0
    else:
        return 1
        
st.set_page_config(
    
    page_title = 'Titanic ML',
    page_icon = ':ship:',
    layout="wide"
    )


train_url = 'https://raw.githubusercontent.com/fowardelcac/Titanic_kaggle_dataset/main/trainn.csv'
test_url = 'https://raw.githubusercontent.com/fowardelcac/Titanic_kaggle_dataset/main/testt.csv'
train, test = pd.read_csv(train_url), pd.read_csv(test_url)

X = train[['Pclass', 'Sex', 'Age']]
y = train.Survived.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

model = modelo(X_train, X_test, y_train, y_test)
st.session_state.rf = modelo
X_test_kaggle = test[['Pclass', 'Sex', 'Age']]
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))    

st.title("Modelo para predecir sobrevivientes del Titanic.")
st.subheader("Esta aplicacion surge a raiz de la competencia de Kaggle.")


    #'Pclass', 'Sex', 'Age'
P_class = st.selectbox("Selecciona tu clase social: ", ('Baja', 'Media', 'Alta'))
Sex_ = st.selectbox("Selecciona tu sexo: ", ('Hombre', 'Mujer'))
Age_ = st.slider("Selecciona tu edad: ", min_value = 1, max_value = 80)

X_predict = [[convert_pclas(P_class), convert_s(Sex_), Age_]]

prediccion = model.predict(X_predict)

if prediccion[0] == 0:
    st.text("No sobreviviente 😔😔")
else:
    st.text("Sobreviviente!! 😃😃")
