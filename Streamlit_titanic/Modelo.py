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

st.title("Visualizaciones.")    
st.text("A continuacion se presentan graficos para generar una mejor compresion de los datos.")

st.subheader("Supervivientes en funcion del sexo")

train = train.rename(columns={'Survived': 'Supervivientes'})

sup_sex = train.groupby('Sex')['Supervivientes'].sum()
sup_sex_df = pd.DataFrame(sup_sex)
sup_sex_df = sup_sex_df.rename(index={0: 'Hombre', 1: 'Mujer'})
st.bar_chart(sup_sex_df)

st.subheader("Supervivientes en funcion de su edad.")
sup_age = train.groupby('Age')['Supervivientes'].sum()
sup_age_df = pd.DataFrame(sup_age)
#st.bar_chart(sup_age_df)
fig = sns.histplot(data=sup_age_df, x= sup_age_df.index, bins = 10)
matplot_fig = fig.get_figure()
st.pyplot(matplot_fig)

st.subheader("Supervivientes en funcion de su clase social.")
sup_clas = train.groupby('Pclass')['Supervivientes'].sum()
sup_clas_df = pd.DataFrame(sup_clas)
sup_clas_df = sup_clas_df.rename(index={1: 'Alta', 2: 'Media', 3: 'Baja'})
st.bar_chart(sup_clas_df)


st.markdown(''' 
  A modo de conclusión, los gráficos presentados nos brindan información valiosa sobre la tragedia del Titanic y la composición de sus pasajeros. En primer lugar, se confirma que el protocolo de "Mujeres y niños primero" fue implementado en la evacuación del barco, ya que el 74.68% de los supervivientes fueron mujeres. Además, se puede observar que los niños menores de 10 años tuvieron una mayor tasa de supervivencia.

Por otro lado, la distribución de los supervivientes según la clase social muestra que la clase acomodada fue la que contó con mayor cantidad de supervivientes. Esto se debe en gran parte a que, al ser una clase social de mayor poder adquisitivo, contaba con mayores privilegios y accesos a las balsas salvavidas. La clase media, por otro lado, fue la que tuvo menor cantidad de supervivientes, lo cual podría deberse a su menor representación en el barco y por ende, menor cantidad de botes salvavidas.

En cuanto a la distribución de pasajeros según la clase social, se observa que la clase baja era la más numerosa, lo cual es un hecho relevante ya que contradice la idea inicial de que el Titanic era un barco exclusivo para ricos. 

En conclusión, los gráficos nos brindan información valiosa para entender la tragedia del Titanic y las diferentes variables que influyeron en la tasa de supervivencia de sus pasajeros.
''')
