# Databricks notebook source
# MAGIC %md #Preparación de datos - Data Preparation

# COMMAND ----------

# MAGIC %md
# MAGIC <p><strong>Objetivo: </strong> El objetivo de este cuaderno es aprender a aplicar diferentes técnicas para obtener la vista minable que será utilizada como entrada a un algoritmo de analítica utilizando el lenguaje de programación Python en Databricks.  </p>

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Vista Minable</h2>

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Indice</h2>
# MAGIC 
# MAGIC <div class="alert alert-block alert-info" style="margin-top: 20px">
# MAGIC <ul>
# MAGIC   <li>Cargar y limpiar los datos</li>
# MAGIC   <li>Normalización</li>
# MAGIC   <li>Discretización</li>
# MAGIC   <li>Numerización</li>
# MAGIC   <li>Agregar variables derivadas</li>
# MAGIC   <li>Oversampling y Undersampling</li>
# MAGIC   <li>Anonimización</li>
# MAGIC </ul>
# MAGIC 
# MAGIC Tiempo estimado: <strong>30 min</strong>
# MAGIC 
# MAGIC </div>
# MAGIC <hr>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Cargar y limpiar los datos

# COMMAND ----------

# MAGIC %md
# MAGIC Carque el mismo conjunto de datos, Automobile, que se utilizó en el cuaderno anterior, utilizando la librería Pandas:

# COMMAND ----------

# Importar libreria requerida
import pandas as pd
# Read data from CSV file
csv_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(csv_path,sep=",",header= None)
# crear la lista headers 
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers
# Imprimer las primeras cinco filas de un dataframe para probar que todo ok
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Realice algunos procesos de limpieza necesarios para continuar con el proceso:

# COMMAND ----------

import numpy as np

# reemplazar "?" por NaN
df.replace("?", np.nan, inplace = True)
#Calcular el promedio de la columna
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
#Remplazar "NaN por el valor de la media en la columna "normalized-losses"
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
#Reemplazamos los valores faltantes en 'num-of-doors' con el valor más frecuente o la moda
df["num-of-doors"].replace(np.nan, "four", inplace=True)
#Remplaza "NaN" por el valor de la media
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)
#Convertir el tipo de datos al formato apropiado
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df["horsepower"]=df["horsepower"].astype(int, copy=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Normalización

# COMMAND ----------

# MAGIC %md
# MAGIC <p>La normalización es el proceso de transformar los valores de algunas variables dentro de un rango similar.</p>
# MAGIC <p>Para hacer una demostración, se normaliza la variable <b>length</b> con normalización MIN-MAX y la variable <b>width</b> con normalización Z-SCORE</p>

# COMMAND ----------

#Normalización MN-MAX. Sustituimos el valor directamente en la columna "length"
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#df[['length','width']] = scaler.fit_transform(df[['length','width']])
df[['length']] = scaler.fit_transform(df[['length']])
df.head()

# COMMAND ----------

#Normalización Z-SCORE. Sustituimos el valor directamente en la columna "width"
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['width']] = scaler.fit_transform(df[['width']])
df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-success alertsuccess" style="margin-top: 20px">
# MAGIC    <strong>AHORA TÚ: </strong>  
# MAGIC   <p>
# MAGIC De acuerdo al ejemplo anterior, normalize la columna "height" con cualquiera de los dos métodos:
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# Escribe tu código aquí y presiona Shift+Enter para ejecutar


# COMMAND ----------

# MAGIC %md
# MAGIC Se muestran las tres columnas normalizadas:

# COMMAND ----------

#Se muestran las tres columnas normalizadas
df[["length","width","height"]].head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Discretización

# COMMAND ----------

# MAGIC %md
# MAGIC <p>En el conjunto de datos, <b>horsepower</b> es una variable con valor en el rango de 48 a 288, tiene 57 valores únicos.¿Qué pasaría si se se observan las diferencias de precio entre automoviles con altos, medios y bajos caballos de fuerza (3 tipos)? ¿Se pueden reacomodar dentro de tres 'grupos' para facilitar el análisis? </p>
# MAGIC 
# MAGIC <p>Se utiliza el método de Pandas <code>cut</code> para discretizar la columna <b>horsepower</b> en 3 grupos.</p>
# MAGIC <p>Se quiere utilizar Discretización por intervalos de igual rango</p>
# MAGIC <p>Observe la variable antes de discretizar</p>

# COMMAND ----------

df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Grafique el histograma de los caballos de fuerza para ver la apariencia de su distribución antes de discretizar:

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC import matplotlib as plt
# MAGIC from matplotlib import pyplot
# MAGIC plt.pyplot.hist(df["horsepower"])
# MAGIC 
# MAGIC # establece las etiquetas x/y y muestra el título 
# MAGIC plt.pyplot.xlabel("horsepower")
# MAGIC plt.pyplot.ylabel("count")
# MAGIC plt.pyplot.title("horsepower bins")

# COMMAND ----------

# MAGIC %md
# MAGIC Paso 1: Crear los intervalos, para este ejemplo serían 3 intervalos de igual rango. Para ello se utiliza la función <code>np.linspace</code>, que divide un rango de valores, en n intervalos equidistantes.

# COMMAND ----------

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins

# COMMAND ----------

# MAGIC %md
# MAGIC Paso 2: Ahora se van a establecer los nombres de los grupos:

# COMMAND ----------

group_names = ['Bajo', 'Medio', 'Alto']

# COMMAND ----------

# MAGIC %md
# MAGIC Paso 3: Se aplica la función <code>cut</code> para determinar a quien pertenece cada valor de <code>df['horsepower']</code>.

# COMMAND ----------

#Le paso a la función la columna donde quiero discretizar, los bins que deseo hacer, los nombres de los grupos y que SI deseo incluir el número más bajo.
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC Grafique el histograma de los caballos de fuerza para ver la apariencia de su distribución despúes de discretizar:

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC import matplotlib as plt
# MAGIC from matplotlib import pyplot
# MAGIC pyplot.bar(group_names, df["horsepower-binned"].value_counts())
# MAGIC 
# MAGIC # establece las etiquetas x/y y muestra el título 
# MAGIC plt.pyplot.xlabel("horsepower")
# MAGIC plt.pyplot.ylabel("count")
# MAGIC plt.pyplot.title("horsepower bins")

# COMMAND ----------

# MAGIC %md
# MAGIC Ejemplo para crear segmentos o grupos personalizados por el analista:

# COMMAND ----------

# EJEMPLO NO EJECUTAR
#df['Age-binned']=pd.cut(x = df['Age'],
                        #bins = [0,18,30,50,120], 
                        #labels = ["Joven", "Adulto Joven", "Adulto","Adulto Mayor"])
#df[['Age','Age-binned']].head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Numerización

# COMMAND ----------

# MAGIC %md
# MAGIC <p>Numerizar una columna que tiene solo dos valores posibles, usando 0 y 1. Se utiliza la variable <b>aspiration</b> donde "std" será 0, mientras "turbo" será 1.</p>
# MAGIC <p>Observe la columna <b>aspiration</b> antes de ser numerizada</p>

# COMMAND ----------

df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC <p>Ahora se aplica la numerización, donde se generan dos colmumnas, pero solo se conversa la primera de las dos, siendo esta suficiente</p>

# COMMAND ----------

#la función get_dummies nos genera una columna para cada categoría de la variable con 0 y 1 si el valor está o no está.
#El parámetro drop_first nos deja una sola columna
df = pd.get_dummies(df, columns = ["aspiration"], drop_first = True)
df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Cuente los valores en la nueva columna:

# COMMAND ----------

df["aspiration_turbo"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Numerizar una columna categórica y ordinal, que tiene varias categorías. Se utiliza la variable <b>num-of-cylinders</b> que es nominal y ordinal. Se muestran los valores actuales de la variable y sus cantidades:

# COMMAND ----------

df["num-of-cylinders"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Se numeriza con la función <code>replace</code> asignando el valor deseado a cada categoría:

# COMMAND ----------

df["num-of-cylinders"].replace({"two":"2","three":"3","four":"4","five":"5","six":"6","eight":"8","twelve":"12"}, inplace = True)
df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Se puede revisar los nuevos valores de la columna y sus cantidades con la función <code>value_counts()</code> y revise el tipo de datos de la nueva columna con la función <code>dtypes</code>:

# COMMAND ----------

#Revisemos nuevamente los valores y sus cantidades para validar que se reemplazaron correctamente:
df["num-of-cylinders"].value_counts()

# COMMAND ----------

#Si miramos el tipo de dato de la columna sigue siendo un Object no olvidar cambiar su tipo de datos
df[['num-of-cylinders']].dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-success alertsuccess" style="margin-top: 20px">
# MAGIC    <strong>AHORA TÚ: </strong>  
# MAGIC   <p>
# MAGIC     Numerize la columna <b>body-style</b> que es categórica y nominal generando columnas para cada uno de los valores:
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

df["body-style"].value_counts()

# COMMAND ----------

# Escribe tu código aquí y presiona Shift+Enter para ejecutar


# COMMAND ----------

# MAGIC %md
# MAGIC Haz doble clic <b>aquí</b> para ver la solución.
# MAGIC 
# MAGIC <!-- Respuesta::
# MAGIC 
# MAGIC df = pd.get_dummies(df, columns = ["body-style"])
# MAGIC df.tail(10)
# MAGIC 
# MAGIC -->

# COMMAND ----------

# MAGIC %md
# MAGIC ##Agregar variables derivadas

# COMMAND ----------

# MAGIC %md
# MAGIC Calculando una nueva variable a partir de combinar otras variables ya existentes. Calculo una variable de tipo 1, el Volumen del Auto, multiplicando las variables <b>length, width y height</b> este calculo debería hacerlo antes de normalizar las variables:

# COMMAND ----------

df['volume'] = df['length']*df['width']*df['height']
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Oversampling y Undersampling

# COMMAND ----------

# MAGIC %md
# MAGIC Cuando se tienen variables desbalanceadas se puede aplicar sobremuestreo o submuestreo de acuerdo a los datos que se tengan. Esta técnica se aplica especialmente sobre variables objetivo de predicción. A manera de ejemplo se utilizará la variable <b>engine-location</b> que está desbalanceada,para demostrar cómo se puede realizar el proceso de submuestreo con Python. Como primer paso se grafica la variable para ver su desbalance en las dos categorías que posee:

# COMMAND ----------

#Grafico de barras para la variable engine-location, se cuentan los valores y se grafican
target_count = df["engine-location"].value_counts()
target_count.plot(kind='bar', title='Count (Engine Location)')

# COMMAND ----------

# MAGIC %md
# MAGIC En este paso se guarda en dos variables separadas la cantidad de cada clase. Las clases son <b>front</b> y <b>rear</b>:

# COMMAND ----------

# Class count
count_class_front, count_class_rear = df["engine-location"].value_counts()
print(count_class_front)
print(count_class_rear)

# COMMAND ----------

# MAGIC %md
# MAGIC Se dividen los ejemplos en el dataframe en dos utilizando las clases como partición:

# COMMAND ----------

# Divide by class
df_class_front = df[df["engine-location"] == "front"]
df_class_rear = df[df["engine-location"] == "rear"]

# COMMAND ----------

# MAGIC %md
# MAGIC Ahora se utilizar el método <code>df.sample</code> para obtener muestras aleatorias de cada clase. En este caso, hay tres ejemplos de la clase <b>rear</b> por lo que si se aplica undersampling, se seleccionan 3 ejemplos aleatorios de la clase <b>front</b>:

# COMMAND ----------

#Ramdom under-sampling. Selecciona de manera aleatoria la misma cantidad de ejemplos que hay de "rear".
df_class_front_under = df_class_front.sample(count_class_rear)
df_class_front_under.head()


# COMMAND ----------

# MAGIC %md
# MAGIC Finalmente se unen en un solo dataframe los dos conjuntos de ejemplos, en este caso sería un total de 6 ejemplos:

# COMMAND ----------

#Unirlos todos en un mismo dataframe
df_test_under = pd.concat([df_class_front_under, df_class_rear], axis=0)
df_test_under

# COMMAND ----------

# MAGIC %md
# MAGIC Si se grafica el dataframe resultante, se puede apreciar que ahora las clases están balanceadas:

# COMMAND ----------

df_test_under["engine-location"].value_counts().plot(kind='bar', title='Count (Engine Location)')

# COMMAND ----------

# MAGIC %md
# MAGIC ##Anonimización

# COMMAND ----------

# MAGIC %md
# MAGIC Existen diversas técnicas de anonimizar y ocultar los datos que se están manjeando en un conjunto de datos. A efectos de demostración se va a utilizar el atributo <b>make</b> que es la marca del carro para ocultar su contenido. Observe cómo se encuentran los datos inicialmente para la Marca.

# COMMAND ----------

df.head(10)

# COMMAND ----------

df_makes_count = df['make'].value_counts()
df_makes_count

# COMMAND ----------

# MAGIC %md
# MAGIC A continuación, se va a codificar make con valores numéricos. Utilizando 0, 1, 2, 3... sucesivamente para cada categoría. La clase LabelEncoder() hace la mayor parte del trabajo por nosotros.

# COMMAND ----------

# MAGIC %md
# MAGIC Es necesario instalar la librería sklearn-pandas para utilizar la clase LabelEncoder:

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install sklearn-pandas

# COMMAND ----------

# MAGIC %md
# MAGIC Se importan las librerías necesarias. <code>DataFrameMapper</code> viene de <code>sklearn_pandas</code> y recibe una lista de elementos a codificar y los nombres de las columnas a transformar.

# COMMAND ----------

# Se importan las librerías necesarias
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
# Se configura la columna a codificar
encoders = [(["make"], LabelEncoder())]
# Se realiza el mapeo de las tuplas para la columna seleccionada
mapper = DataFrameMapper(encoders, df_out=True)
# Se une la columna anonimizada con el resto del dataframe
new_cols = mapper.fit_transform(df.copy())
df = pd.concat([df.drop(columns=["make"]), new_cols], axis="columns")

# COMMAND ----------

# MAGIC %md
# MAGIC Compruebe cómo ahora la columna make no muestra los nombres de las marcas sino números.

# COMMAND ----------

df_makes_count = df['make'].value_counts()
df_makes_count

# COMMAND ----------

# MAGIC %md
# MAGIC ##Links de ayuda interesantes
# MAGIC <ul>
# MAGIC   <li>Resampling strategies for imbalanced dataset: https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets</li>
# MAGIC   <li>A simple way to anonymize data with Python and Pandas: https://dev.to/r0f1/a-simple-way-to-anonymize-data-with-python-and-pandas-79g</li>
# MAGIC <ul>
