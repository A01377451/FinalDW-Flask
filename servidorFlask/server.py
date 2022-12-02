from flask import Flask, request, jsonify, render_template
import numpy as np
from joblib import load
from werkzeug.utils import secure_filename
import os
import json
import pandas as pd
from joblib import load, dump

import requests

# Cargar el modelo
dt = load('modelo.joblib')

# Generar el servidor (Back-end)
TEMPLATE_DIR = os.path.abspath('../templates')
STATIC_DIR = os.path.abspath('../static')
servidorWeb = Flask(__name__)

# Formulario
@servidorWeb.route("/modeloPrediccion",methods=['GET'])
def formulario():
    return render_template('formulario.html')


# Predicción de datos
@servidorWeb.route('/modeloForm', methods=['POST'])
def modeloForm():
    #Procesar datos de entrada 
    contenido = request.form
    print(contenido)
    datosEntrada = np.array([
            contenido['SepalLengthCmP'],
            contenido['SepalWidthCmP'],
            contenido['PetalLengthCmP'],
            contenido['PetalWidthCmP']
        ])
    #Utilizar el modelo
    resultado=dt.predict(datosEntrada.reshape(1,-1))
    #Regresar la salida del modelo
    return jsonify({"Resultado":str(resultado[0])})

# Reentrenar modelo
@servidorWeb.route('/reEntrenar', methods=['POST'])
def reEntrenar():
    url = 'http://localhost:8082/flor/consultarFlores'
    r = requests.get(url = url)
    inputs = r.json()
    
    add = []
    headers = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm',"Species"]
    
    dataFrame = pd.read_csv("Iris.csv")
    

    for i in range(len(inputs)):
        add=[]
        add.append(len(dataFrame.index)+1)
        for j in range(len(headers)):
            if j == len(headers)-1:
                add.append(inputs[i][headers[j]])
            else:
                add.append(int(inputs[i][headers[j]]))
        print(add)
        dataFrame.loc[len(dataFrame.index)]=add

    

    # Características de entrada (Información de los campos del formulario)
    X=(dataFrame.drop('Species', axis=1)).drop('Id', axis=1) #axis=0 saca columnas

    # Cracterísticas de salida ()
    y=dataFrame['Species']

    # Separar la base de datos en 2 conjuntos
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)
    
    #Modelo
    from sklearn.tree import DecisionTreeClassifier
    dt=DecisionTreeClassifier()

    #Entrenar modelo
    dt.fit(X_train,y_train)

    # Exportar el modelo para usarlo en un servidor web con flask
    dump(dt,'../modelo.joblib')  # 64 bits

    #Regresar la salida del modelo
    return jsonify({"result": str(dt.score(X_test, y_test))})


# Predicción de datos a través de JSON
@servidorWeb.route('/modelo', methods=['POST'])
def modelo():
    #Procesar datos de entrada 
    contenido = request.json
    print(contenido)
    datosEntrada = np.array([
            contenido['SepalLengthCm'],
            contenido['SepalWidthCm'],
            contenido['PetalLengthCm'],
            contenido['PetalWidthCm']
        ])
    #Utilizar el modelo
    resultado=dt.predict(datosEntrada.reshape(1,-1))
    #Regresar la salida del modelo
    return jsonify({"Resultado":str(resultado[0])})

if __name__ == '__main__':
    servidorWeb.run(debug=False,host='0.0.0.0',port='8081')
