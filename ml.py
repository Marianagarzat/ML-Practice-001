# PRIMER EJEMPLO - APRENDIZAJE SUPERVISADO CON REGRESIÓN LINEAL

from sklearn.linear_model import LinearRegression #sklearn contiene herramientas de ML como LinearRegression...

import numpy as np # permite trabajar con vectores, arreglos numericos, matrices, etc)


 #---------------------------------------------------#


# Datos de ENTRENAMIENTO por aprendizaje supervisado (nosotros le indicamos si es correcto o no, etc)

# Inversion de publicidad (miles de pesos)
x = np.array([1,2,3,4,5,6]).reshape(-1,1) # variables independientes

#El reshape (-1,1) te convierte los numeros a columnas

# reshape convierte el arreglo de una dimension, en una dimension

"""[[1],
[2],
[3]] """ # asi te lo convierte

# porque el modelo de regresion lineal,  espera una matriz de dos dimensiones

# ventas obtenidas en miles de pesos
y = np.array([3,5,7,9,11,13]) # variable dependiente que queremos predecir


# si invertimos 1 pesos, obtenemos 3. Si invertimos 2, obtenemos 5 (checa la x contra y)

# Por esto es supervisado, porque ya sabemos cuanto nos da todo

# creamos ahora el MODELO

modelo = LinearRegression() #creamos un objeto

# Ahora LO ENTRENAMOS!

modelo.fit(x,y) #creamos modelo con los datos es como y= mx +b


# vamos a hacer una proyeccion

nueva_inversion = np.array([[8]]) #esta es la x, o sea los miles invertidos
prediccion = modelo.predict(nueva_inversion) # queremos estimar las ventas para una inversion de 8mil pesos de publicidad


# Mostramos resultados

print("Pendiente: ", modelo.coef_[0])
print("Intercepto: ", modelo.intercept_)
print("Ventas estimadas: ", prediccion[0]," mil pesos")


# si la pendiente es positiva indica que si metemos publicidad, tendremos una pendiente positiva.

# Podemos graficar esto tmb