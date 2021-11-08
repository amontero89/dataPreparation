# Databricks notebook source
# MAGIC %md #Preparación de datos - Data Preparation

# COMMAND ----------

# MAGIC %md <p><strong>Objetivo: </strong> El objetivo de este cuaderno es aprender a limpiar y corregir los problemas de calidad de los datos identificados en la exploración utilizando el lenguaje de programación Python en Databricks.  </p> 

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Limpieza de Datos</h2>

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Indice</h2>
# MAGIC 
# MAGIC <div class="alert alert-block alert-info" style="margin-top: 20px">
# MAGIC <ul>
# MAGIC   <li>Cargar los Datos</li>
# MAGIC   <li>Tratar datos perdidos</li>
# MAGIC   <li>Cambiar tipos de datos de las columnas</li>
# MAGIC   <li>Identificar atributos que todos son iguales o todos son diferentes</li>
# MAGIC   <li>Eliminar columnas</li>
# MAGIC   <li>Eliminar filas</li>
# MAGIC   <li>Identificar y tratar posibles datos atípicos</li>
# MAGIC   <li>Matriz de correlación</li>
# MAGIC </ul>
# MAGIC 
# MAGIC Tiempo estimado: <strong>30 min</strong>
# MAGIC 
# MAGIC </div>
# MAGIC <hr>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Cargar los datos

# COMMAND ----------

# MAGIC %md
# MAGIC Carque los mismos datos que se utilizaron en el cuaderno anterior, utilizando la librería Pandas:

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
# MAGIC ##Tratar datos perdidos

# COMMAND ----------

# MAGIC %md
# MAGIC Como se puede ver, algunos signos de interrogación aparecen en el dataframe; estos son valores faltantes que pueden dificultar el futuro análisis.
# MAGIC ¿Como identificar y manejar todos aquellos valores que faltan?
# MAGIC ¿Como trabajar con valores faltantes?

# COMMAND ----------

# MAGIC %md
# MAGIC Revisar los datos perdidos con la función <code>print(df.isnull().sum())</code>

# COMMAND ----------

print(df.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC En el conjunto de datos del automovil, los datos que faltan aparecen con el signo "?".
# MAGIC Cambie "?" por NaN (Not a Numer) el cual es el marcador por defecto de Python para valores faltantes por razones de conveniencia y velocidad de computo. Aqui se utiliza la función <code>replace()</code>

# COMMAND ----------

import numpy as np

# reemplazar "?" por NaN
df.replace("?", np.nan, inplace = True)
df.replace("l", np.nan, inplace = True)
df.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC Si desea visualizar los registros que contienen datos perdidos puede ejecutar el siguiente código:

# COMMAND ----------

# Para ver todas las filas que tienen valores faltantes
is_NaN = df.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = df[row_has_NaN]
rows_with_NaN.head(50)

# COMMAND ----------

# MAGIC %md
# MAGIC <h5>Resolviendo los problemas con los datos faltantes: </h5>

# COMMAND ----------

# MAGIC %md
# MAGIC <p>Reemplazar con la media</p>
# MAGIC La variable <b>normalized-losses</b> tiene 41 datos faltantes. Revisar la simetría del atributo y luego decidir si se utilza media o mediana:

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(df["normalized-losses"])

# COMMAND ----------

df['normalized-losses'].hist(figsize = (6,6))
plt.show

# COMMAND ----------

mean = df['normalized-losses'].mean()
median = df['normalized-losses'].median()
mode = df['normalized-losses'].mode()
skew = df['normalized-losses'].skew()
kurt = df['normalized-losses'].kurt()
print("La media es:", mean)
print("La mediana es:", median)
print("La moda es:", mode)
print("El sesgo es:", skew)
print("La kurtosis es:", kurt)

# COMMAND ----------

#Calcular el mediana de la columna
m_norm_loss = df["normalized-losses"].astype("float").median(axis=0)
#Remplazar "NaN por el valor de la mediana en la columna "normalized-losses"
df["normalized-losses"].replace(np.nan, m_norm_loss, inplace=True)

# COMMAND ----------

df.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC <p>Remplazar con la Frecuencia:</p>
# MAGIC La variable <b>num-of-doors</b> tiene 2 datos faltantes, remplazar con "four". Razón: El 84% de los sedanes es de cuatro puertas. Debido que tener cuatro puertas es mas probable que ocurra.

# COMMAND ----------

#Para ver los valores presentes en una columna podemos usar el método ".value_counts()":
df['num-of-doors'].value_counts()

# COMMAND ----------

#Podemos ver que el tipo mas común es el de cuatro puertas. Además podemos usar el método .idxmax()" para calcular automaticamente el tipo mas comun:
df['num-of-doors'].value_counts().idxmax()

# COMMAND ----------

#Reemplazamos los valores faltantes en 'num-of-doors' con el valor más frecuente o la moda
df["num-of-doors"].replace(np.nan, "four", inplace=True)

# COMMAND ----------

print(df.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC <p>Eliminar filas</p>
# MAGIC <p>Eliminar todas las filas que tienen datos faltantes en la columna <b>price</b></p>

# COMMAND ----------

#Elimina toda la fila con NaN en la columna "price"
# df_new = df.dropna(subset=["price"], axis=0)
df.dropna(subset=["price"], axis=0, inplace=True)

#Restablece el índice debido a que eliminamos dos filas
df.reset_index(drop=True, inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-success alertsuccess" style="margin-top: 20px">
# MAGIC    <strong>AHORA TÚ: </strong>  
# MAGIC   <p>
# MAGIC Calcula el valor de la media para la columna 'peak-rpm' y remplaza "NaN" por el valor de la media:
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# Escribe tu código aquí y presiona Shift+Enter para ejecutar


# COMMAND ----------

# MAGIC %md
# MAGIC Haz doble clic <b>aquí</b> para ver la solución.
# MAGIC 
# MAGIC <!-- Respuesta::
# MAGIC 
# MAGIC avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
# MAGIC df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)
# MAGIC 
# MAGIC -->

# COMMAND ----------

# MAGIC %md
# MAGIC Compruebe el resultado final:

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Cambiar tipos de datos de las columnas

# COMMAND ----------

# MAGIC %md
# MAGIC Un paso importante en la limpieza de datos es asegurarse que el formato de cada columna sea el correcto (int, float, text u otro)

# COMMAND ----------

#Listar los tipos de datos para cada columna
df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC <p>Como podemos ver arriba, algunas columnas no tienen el tipo correcto de dato. Las variables numéricas deben ser de tipo 'float' o 'int', y las variables con cadenas como pueden ser las categorias deben ser de tipo 'object'. Tome un minuto para analizar los tipos de datos de cada columna, puede comparar con el diccionario de datos compartido en la clase.</p>

# COMMAND ----------

#Convertir el tipo de datos al formato apropiado
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-success alertsuccess" style="margin-top: 20px">
# MAGIC    <strong>AHORA TÚ: </strong>  
# MAGIC   <p>
# MAGIC     Cambie el valor de la columna <b>peak-rpm</b> a tipo <b>float</b>
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# Escribe tu código aquí y presiona Shift+Enter para ejecutar


# COMMAND ----------

# MAGIC %md
# MAGIC Haz doble clic <b>aquí</b> para ver la solución.
# MAGIC 
# MAGIC <!-- Respuesta::
# MAGIC 
# MAGIC df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
# MAGIC 
# MAGIC -->

# COMMAND ----------

# MAGIC %md
# MAGIC Vuelva a ejecutar la función <code>df.dtypes</code> para validar el cambio.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Identificar atributos que todos son iguales o todos son diferentes

# COMMAND ----------

# MAGIC %md
# MAGIC Para identificar variables donde todos sus valores sean iguales, primero se pueden analizar las variables categóricas. La configuración predeterminada de la funcion <code>describe</code> omite las variables de tipo objeto. Podemos aplicar la función en las variables de tipo ‘objeto’ de la siguiente manera:

# COMMAND ----------

#unique, top y frequency ("único, superior y frecuencia").
df.describe(include="object")

# COMMAND ----------

df['engine-location'].value_counts()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
sns.catplot(x="engine-location", kind="count", palette="rocket", data=df, height = 4, aspect = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Eliminar columnas

# COMMAND ----------

# MAGIC %md
# MAGIC Para eliminar una columna, o varias columnas, use el nombre de la columna y especifique el "eje". Eliminemos la columna <b>engine-location</b> pues no aporta a la predicción del precio.

# COMMAND ----------

#Si utilizamos inplace=True los cambios se aplican directamente sobre el dataframe
df.drop("engine-location", axis=1, inplace=True)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Otras alternativas para eliminar columnas son:
# MAGIC <p>Si no quiero perder la columna de los datos originales, asigno a un nuevo dataframe:</p>
# MAGIC <code>new_df = df.drop("Column_Name", axis=1)</code>
# MAGIC <p>Si quiero eliminar varias columnas:</p>
# MAGIC <code>new_df = df.drop(["Column_Name_1","Column_Name_2","Column_Name_N"], axis=1)</code>

# COMMAND ----------

# MAGIC %md
# MAGIC ###Eliminar filas

# COMMAND ----------

# MAGIC %md
# MAGIC Las filas también se pueden eliminar utilizando la función <code>drop</code>, especificando <code>axis = 0</code>.

# COMMAND ----------

# MAGIC %md
# MAGIC Podemos eliminar una fila específica que deseemos eliminar, ya sea porque tiene datos faltantes o datos atípicos. A manera de prueba eliminemos la fila con índices 0 y 1, pero sin afectar el dataframe original.

# COMMAND ----------

delete_rows = df.drop([0,1], axis=0)
delete_rows.head(4)

# COMMAND ----------

# MAGIC %md
# MAGIC Si desemos mantener el Indice (el índice funciona como un label, id o tag de las filas) podemos utilizar la función <code>reset_index</code>

# COMMAND ----------

#Actualización del index
delete_rows.reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Identificar y tratar posibles datos atípicos

# COMMAND ----------

# MAGIC %md
# MAGIC La primera herramienta que se puede aplicar para identificar outliers, son los gráficos de BoxPlot. Se muestra el atributo <b>length</b> con un gráfico de BoxPlot.

# COMMAND ----------

sns.boxplot(df["length"], orient="v")

# COMMAND ----------

# MAGIC %md
# MAGIC Se puede evaluar cada dato y decidir si efectivamente son datos atípicos o son datos normales solo que extremos.

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-success alertsuccess" style="margin-top: 20px">
# MAGIC    <strong>AHORA TÚ: </strong>  
# MAGIC   <p>
# MAGIC     Elija otra variable y grafique su BoxPlot para identificar datos atípicos:
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# Escribe tu código aquí y presiona Shift+Enter para ejecutar


# COMMAND ----------

# MAGIC %md
# MAGIC Haz doble clic <b>aquí</b> para ver la solución.
# MAGIC 
# MAGIC <!-- Posible respuesta:
# MAGIC 
# MAGIC sns.boxplot(df["highway-mpg"], orient="v")
# MAGIC 
# MAGIC -->

# COMMAND ----------

# MAGIC %md
# MAGIC Lo segundo que se puede hacer es comparar dos atributos y ver si entre ellos surge algún dato atípico. Se muestran los atributos peso <b>curb-weight</b> y millas por galon en autopista <b>highway-mpg</b> con un gráfico de puntos.

# COMMAND ----------

sns.set(style="whitegrid")
sns.regplot(x="curb-weight", y="highway-mpg", data=df)

# COMMAND ----------

# MAGIC %md
# MAGIC Cuando se quiere analizar todas las variables para saber si hay valores atípicos y inconsistentes entre ellas, se puede utilizar el algoritmo de LOF para identificar aquellos registros atípicos:
# MAGIC <p>Utilice la libería ScikitLearn que contiene el algoritmo para LOF, para fines prácticos aplique el algoritmo sobre los atributos numéricos:</p>

# COMMAND ----------

#Importe la librería ScikitLearn específicamente el algoritmo de LOF
from sklearn.neighbors import LocalOutlierFactor

# COMMAND ----------

# MAGIC %md
# MAGIC A continuación se aplica el modelo a las columnas numéricas y aquella que no tienen valores faltantes (el algoritmo no soporta datos faltantes) y se muestran los registros que el algoritmo identifica como inconsistentes:

# COMMAND ----------

#Seleccionar columnas
select_df = df[["symboling","normalized-losses","wheel-base","length","width","height","curb-weight","engine-size"]]

#Especificar el modelo que se va a utilizar
model = LocalOutlierFactor(n_neighbors = 20)

#Ajuste al modelo
y_pred = model.fit_predict(select_df)
y_pred

# COMMAND ----------

#Filtrar los indices de los outliers
outlier_index = (y_pred == -1) #los valores negativos son outliers

#Filtrar los valores de los outliers en el dataframe
outlier_values = select_df.iloc[outlier_index]
outlier_values

# COMMAND ----------

# MAGIC %md
# MAGIC El algoritmo LOF identifica 21 datos inconsistentes en el conjunto de datos.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Atributos redundantes

# COMMAND ----------

# MAGIC %md
# MAGIC Para identificar los atributos redundantes se pueden utilizar la matriz de correlación e indentificar correlaciones entre atributos. Se puede calcular la correlación entre variables de tipo “int64” o “float64” utilizando el método <code>corr</code> de la librería Pandas. Los elementos diagonales tiene siempre valor 1:

# COMMAND ----------

corrMatrix=df.corr()
corrMatrix

# COMMAND ----------

#Importe nuevamente las librias de Seaborn y Matplotlib en caso de que no estén importadas
import seaborn as sns
import matplotlib.pyplot as plt
#Sentencia para ajustar la visualización y tamaño del gráfico
f, ax = plt.subplots(figsize=(10, 8))
#HeatMap de Seaborn, annot:muestra valores, fmt:decimales, ax:visializacion
sns.heatmap(corrMatrix,annot = True,fmt='.1g',ax=ax)

# COMMAND ----------

# MAGIC %md <hr/>
# MAGIC <div class="alert alert-success alertsuccess" style="margin-top: 20px">
# MAGIC    <strong>AHORA TÚ: </strong>  
# MAGIC   <p>Encuentre la correlación entre las siguientes columnas: bore, stroke, compression-ratio , y horsepower y muestre su matriz de correlación:</p>
# MAGIC <p>Pista: si desea seleccionar aquellas columnas utilice la siguiente sintaxis: df[['bore','stroke' ,'compression-ratio','horsepower']]</p>
# MAGIC </div>
# MAGIC <hr/>

# COMMAND ----------

# Escribe tu código aquí y presiona Shift+Enter para ejecutar


# COMMAND ----------

# MAGIC %md
# MAGIC Haz doble clic <b>aquí</b> para ver la solución.
# MAGIC 
# MAGIC <!-- The answer is below:
# MAGIC 
# MAGIC matrix = df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()  
# MAGIC sn.heatmap(matrix,annot = True,fmt='.1g')
# MAGIC 
# MAGIC -->

# COMMAND ----------

# MAGIC %md
# MAGIC ##Links de ayuda interesantes
# MAGIC <ul>
# MAGIC     <li>Documentacion de LOF: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html</li>
# MAGIC     <li>Ajustes a los parámetros de la matriz de correlación: https://heartbeat.fritz.ai/seaborn-heatmaps-13-ways-to-customize-correlation-matrix-visualizations-f1c49c816f07</li>
# MAGIC <ul>
