import streamlit as st


st.set_page_config(
    
    page_title = 'Explicacion',
    page_icon = ':Book:',
    layout="wide"
    )

st.title("Breve explicacion.")
st.markdown('''
            Esta aplicacion utiliza un algoritmo de Machine Learning llamado 'Random Forest', con el   objetivo predecir si un pasajero del Titanic sobrevivió o no, 
            en función de tres atributos: edad, clase social y sexo. EL algoritmo random forest es una técnica de aprendizaje supervisado que combina múltiples árboles 
            de decisión para generar una predicción. En este caso, el modelo utiliza tres atributos para construir múltiples árboles de decisión que predicen la supervivencia
            de los pasajeros. Los atributos utilizados son la edad del pasajero, la clase social a la que pertenece (alta, media o baja) y el sexo (hombre o mujer). 
            Estos atributos se consideran importantes para predecir la supervivencia de los pasajeros, ya que la edad y la clase social pueden indicar la capacidad de 
            acceso a recursos y la vulnerabilidad, mientras que el género puede influir en el acceso a botes salvavidas. El modelo ha logrado una precisión del 78% en el
            conjunto de datos de entrenamiento. Esto significa que el modelo puede predecir correctamente si un pasajero sobrevivió o no en un 78% de los casos.
            
            Los datos fueron obtenidos de la competencia de Kaggle sobre el dataset del titanic.            
    
            
            ''')