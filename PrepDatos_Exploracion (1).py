# Databricks notebook source
# MAGIC %md 
# MAGIC #Preparación de datos - Data Preparation

# COMMAND ----------

# MAGIC %md <p><strong>Objetivo: </strong> El objetivo de este cuaderno es cargar y realizar la exploración inicial de los datos utilizando el lenguaje de programación Python en Databricks.  </p> 

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Exploración de Datos</h3>

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Indice</h2>
# MAGIC 
# MAGIC <div class="alert alert-block alert-info" style="margin-top: 20px">
# MAGIC <ul>
# MAGIC   <li>Cargar los Datos</li>
# MAGIC   <li>Visualizar los Datos</li>
# MAGIC   <li>Tipos de datos</li>
# MAGIC   <li>Visualizar las estadísticas</li>
# MAGIC   <li>Identificar datos faltantes</li>
# MAGIC   <li>Explorar relaciones entre los datos</li>
# MAGIC   <li>Graficar las estadísticas</li>
# MAGIC   <li>Exportar los datos</li>
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
# MAGIC Existen varios formatos para un conjunto de datos, .csv, .json, .xlsx, etc. Los datos pueden ser almacenados en distintos lugares, ya sea localmente o en línea.<b>
# MAGIC En estas sección aprenderá a cargar un conjunto de datos en su cuaderno de Databricks.</b>
# MAGIC En nuestro caso el conjunto de datos Automobile es de una fuente en línea en formato CSV (valores separados por coma). Usemos este conjunto como ejemplo para practicar la lectura de datos.
# MAGIC <ul>
# MAGIC     <li>fuente de datos: <a href="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data" target="_blank">https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data</a></li>
# MAGIC     <li>tipo de datos: csv</li>

# COMMAND ----------

# MAGIC %md
# MAGIC Vamos a utilizar la librería Pandas de Python para realizar la lectura de archivos. Le ponemos un alias <strong>pd</strong> para que sea más fácil utilizarla:

# COMMAND ----------

# Importar libreria requerida
import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC Después del comando para importar, ahora tenemos acceso a una gran cantidad de clases y funciones predefinidas. Una forma en que pandas le permite trabajar con datos es con dataframes. Repasemos el proceso para pasar de un archivo de valores separados por comas (<b>.csv</b>) a un dataframe. Esta variable <code>csv_path</code> almacena la ruta de <b>.csv</b>, que se utiliza como argumento para la función <code>read_csv</code>. El resultado se almacena en el objeto <code>df</code>, esta es una forma corta común que se usa para una variable que se refiere a un dataframe de Pandas.

# COMMAND ----------

# Read data from CSV file
csv_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(csv_path,sep=",",header= None)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Visualizar los datos

# COMMAND ----------

# MAGIC %md
# MAGIC Podemos utilizar el método <code>dataframe.head()</code> para examinar las primeras cinco filas del dataframe, se utiliza cuando el conjuntos de datos es muy grande y no queremos cargar todo: 

# COMMAND ----------

# Imprimer las primeras cinco filas de un dataframe
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Después de leer el conjunto de datos podemos utilizar el método <code>dataframe.head(n)</code> para revisar las primeras n filas del dataframe; donde n es un entero. Al contrario de <code>dataframe.head(n)</code>, <code>dataframe.tail(n)</code> mostrará las n filas del final del dataframe.

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-success alertsuccess" style="margin-top: 20px">
# MAGIC    <strong>AHORA TÚ: </strong>  
# MAGIC   <p>
# MAGIC Revise las ultimas 10 filas del dataframe "df":
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# Escribe tu código aquí y presiona Shift+Enter para ejecutar


# COMMAND ----------

# MAGIC %md
# MAGIC Haz doble clic <b>aquí</b> para ver la solución.
# MAGIC 
# MAGIC <!-- Respuesta:
# MAGIC 
# MAGIC print("The last 10 rows of the dataframe\n")
# MAGIC df.tail(10)
# MAGIC 
# MAGIC -->

# COMMAND ----------

# MAGIC %md
# MAGIC <h5 id="func">Añadir cabeceras</h5>

# COMMAND ----------

# MAGIC %md
# MAGIC <p>
# MAGIC Observe el conjunto de datos; Pandas automaticamente establece la cabecera en un entero a partir de 0.  
# MAGIC Para describir mejor nuestros datos podemos agregarle una cabecera, esta información esta disponible en: <a href="https://archive.ics.uci.edu/ml/datasets/Automobile" target="_blank">https://archive.ics.uci.edu/ml/datasets/Automobile</a>
# MAGIC </p>
# MAGIC <p>
# MAGIC De este modo debemos agregar las cabeceras manualmente.    
# MAGIC Primero creamos una lista <b>headers</b> que incluya todos los nombres de columna en orden.
# MAGIC Despues usamos <code>dataframe.columns = headers</code> para reemplazar las cabeceras por la lista que hemos creado.
# MAGIC </p>

# COMMAND ----------

# crear la lista headers 
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)

# COMMAND ----------

# MAGIC %md
# MAGIC Remplazamos las cabeceras y volvemos a revisar nuestro dataframe:

# COMMAND ----------

df.columns = headers
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC <h5>Acceder a una columna y ver sus valores</h5>

# COMMAND ----------

# MAGIC %md
# MAGIC Se accede a una columna especificando el nombre de la misma. Por ejemplo, puedes acceder a la columna <b>symboling</b> y a la columna <b>body-style</b>. Cada una de estas columnas es una serie de Pandas.

# COMMAND ----------

x=df[["symboling"]]
x

# COMMAND ----------

y=df[["body-style"]]
y

# COMMAND ----------

# MAGIC %md
# MAGIC <h5 id="func">Visualizar con databricks</h5>

# COMMAND ----------

# MAGIC %md
# MAGIC Podemos utilizar la funcion <code>display()</code> de Databricks para visualizar la Tabla y graficar los atributos:

# COMMAND ----------

# Esto es código spark
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Tipos de datos

# COMMAND ----------

# MAGIC %md
# MAGIC <p>
# MAGIC Los datos se encuentran en una variedad de tipos.<br>
# MAGIC Los tipos principales almacenados en dataframes de Pandas son <b>object</b>, <b>float</b>, <b>int</b>, <b>bool</b> y <b>datetime64</b>. Para aprender mejor acerca de cada atributo es mejor para nosotros saber el tipo de dato de cada columna.
# MAGIC </p>

# COMMAND ----------

#La función dtypes genera una tabla con el tipo de dato de cada columna
df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC <h5>Tipo de dato de una columna específica</h5>

# COMMAND ----------

# MAGIC %md
# MAGIC De esta forma podemos consultar cuál es el tipo de dato de una columna específica:

# COMMAND ----------

#Separamos la columna en una dataframe llamado df_column
df_column=df[['engine-size']]
df_column.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC <h5>Cambiar el tipo de dato de una columna específica</h5>

# COMMAND ----------

# MAGIC %md
# MAGIC ¿Cómo cambiar el tipo de dato de una columna específica? Cambiemos el tipo de datos de la columna <b>Price</b> que fue identificado como <b>object</b> y es un <b>float</b>.

# COMMAND ----------

#utilizamos errors='coerce' para ignorar los datos faltantes
df["price"] = pd.to_numeric(df["price"],errors='coerce')
df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC Como se muestra, se observa claramente que el tipo de dato de <b>symboling</b> y <b>curb-weight</b> es <code>int64</code>, <b>normalized-losses</b> es <code>object</code> pero debería ser de tipo numérico, al igual que <b>bore</b>, etc. Estos tipos de datos pueden modificarse.

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-success alertsuccess" style="margin-top: 20px">
# MAGIC    <strong>AHORA TÚ: </strong>  
# MAGIC   <p>
# MAGIC Cambie el tipo de datos de la columna "stroke":
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
# MAGIC df["stroke"] = pd.to_numeric(df["stroke"],errors='coerce')
# MAGIC 
# MAGIC -->

# COMMAND ----------

# MAGIC %md
# MAGIC ##Visualizar las estadísticas

# COMMAND ----------

# MAGIC %md
# MAGIC <p>Este conjunto de datos es pequeño, pero si se quisiera saber la cantidad de atributos y de elementos que se tienen en el conjunto de datos, se puede utilizar la función <code>dataframe.shape</code>. Esta función visualiza primero el número de elementos y luego el número de atributos.</p>

# COMMAND ----------

df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC <p>Vamos a utilizar la función <code>dataframe.describe</code> para visualizar las estadísticas del conjunto de datos. Por defecto, la función <code>dataframe.describe</code> muestra las filas y columnas que contienen números.</p>
# MAGIC Esto mostrará:
# MAGIC <ul>
# MAGIC     <li>el recuento de esa variable</li>
# MAGIC     <li>la media</li>
# MAGIC     <li>la desviación estándar (std)</li> 
# MAGIC     <li>el valor mínimo</li>
# MAGIC     <li>el IQR (rango intercuartil: 25%, 50% y 75%)</li>
# MAGIC     <li>el valor máximo</li>
# MAGIC <ul>

# COMMAND ----------

df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Si se quisiera calcular la mediana de una variable en específico se puede de la siguiente manera:

# COMMAND ----------

#Muestra la mediana para los atributs "length" y "compression-ratio"
median= df[['length', 'compression-ratio']].median()
median

# COMMAND ----------

# MAGIC %md
# MAGIC Por defecto la función solo muestra atributos que son numéricos. Es posible hacer que la función <code>describe</code> funcione también para las columnas de tipo object. Para permitir un resumen de todas las columnas, podríamos añadir un argumento <code>include="all"</code> entre los paréntesis de la función <code>describe</code>.

# COMMAND ----------

#unique, top y frequency ("único, superior y frecuencia").
#df.describe(include="object")
df.describe(include="all")

# COMMAND ----------

# MAGIC %md
# MAGIC <h5>Contar Valores</h5>

# COMMAND ----------

# MAGIC %md
# MAGIC Una forma de resumir los datos categóricos es usando la función <code>value_counts</code>. Por ejemplo, en nuestro conjunto de datos, tenemos el lugar del motor (<b>engine-location</b>) como una variable categórica de frontal y trasero.

# COMMAND ----------

drive_wheels_counts = df['make'].value_counts().to_frame()
drive_wheels_counts

# COMMAND ----------

# MAGIC %md
# MAGIC <p>Examinar los recuentos de valores de la ubicación del motor no sería una buena variable predictiva del precio. Esto se debe a que solo tenemos tres autos con motor trasero y 202 con motor delantero, este resultado es sesgado. Por lo tanto, no podemos sacar ninguna conclusión sobre la ubicación del motor.</p>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-success alertsuccess" style="margin-top: 20px">
# MAGIC    <strong>AHORA TÚ: </strong>  
# MAGIC   <p>
# MAGIC Puede seleccionar las columna de un dataframe indicando el nombre de cada una, por ejemplo, puede seleccionar tres columnas de la siguiente manera:
# MAGIC </p>
# MAGIC <p>
# MAGIC     <code>dataframe[['column 1',column 2', 'column 3']]</code>
# MAGIC </p>
# MAGIC <p>
# MAGIC Donde "column" es el nombre de la columna se puede aplicar el método ".describe()" para obtener las estadísticas de aquellas columnas de la siguiente manera:
# MAGIC </p>
# MAGIC <p>
# MAGIC     <code>dataframe[[' column 1 ',column 2', 'column 3'] ].describe()</code>
# MAGIC </p>
# MAGIC Aplicar el método ".describe()" a las columnas 'length' y 'compression-ratio'.
# MAGIC </div>

# COMMAND ----------

# Escribe tu código aquí y presiona Shift+Enter para ejecutar


# COMMAND ----------

# MAGIC %md
# MAGIC Haz doble clic <b>aquí</b> para ver la solución.
# MAGIC 
# MAGIC <!-- Respuesta:
# MAGIC 
# MAGIC df[['length', 'compression-ratio']].describe()
# MAGIC 
# MAGIC -->

# COMMAND ----------

# MAGIC %md
# MAGIC ##Identificar datos faltantes

# COMMAND ----------

# MAGIC %md
# MAGIC Debemos visualizar nuestros datos e identificar el valor(es) que se está utilizando para los datos faltantes. Los valores faltantes pueden ser espacios vacíos, NA, n/a, --, 0, o cualquier otro valor que no es considerado correcto en esa columna.

# COMMAND ----------

#Identifique cual es el valor que se está utilizando para los datos faltantes en el set de datos:
df.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC Con la función <code>isnull</code> podemos saber cuantos datos faltantes identifica Python en nuestro set de datos.

# COMMAND ----------

print(df.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC Todavía Python no está identificando los datos faltantes en el conjunto de datos, sino que los está tratando como un valor correcto más. Para marcar los datos faltantes se realiza los siguiente:

# COMMAND ----------

#Realice una lista de los valores que son identificados como datos faltantes
#No olvide al final volver a cargar las cabeceras
missing_values = ["?", "l"]
csv_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(csv_path,sep=",",header= None, na_values = missing_values)
df.head(20)

# COMMAND ----------

# crear la lista headers 
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Vuelva a ejectuar la sentencia <code>print(df.isnull().sum())</code> para visualizar los datos faltantes

# COMMAND ----------

print(df.isnull().sum())

# COMMAND ----------

df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ##Explorar relaciones entre los datos

# COMMAND ----------

# MAGIC %md
# MAGIC <h5>Variables numéricas continuas:</h5> 
# MAGIC 
# MAGIC <p>Las variables numéricas continuas son variables que pueden contener cualquier valor dentro de cierto rango. Las variables numéricas continuas pueden tener el tipo <b>int64</b> o <b>float64</b>. Una excelente manera de visualizar estas variables es mediante el uso de diagramas de dispersión con líneas ajustadas.</p>
# MAGIC <p>Para comenzar a comprender la relación (lineal) entre una variable individual y el precio. Podemos hacer esto usando <code>regplot</code>, que traza el diagrama de dispersión más la línea de regresión ajustada para los datos.</p>

# COMMAND ----------

# MAGIC %md
# MAGIC Vamos a importar las librerías Matplotlib y Seaborn para la visualización de datos. Les ponemos un alias <strong>plt</strong> y <strong>sns</strong> para que sea más fácil su uso:

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC El gráfico <code>seaborn.pairplot</code> permite visualizar todas las variables numéricas y la combinación entre ellas. Pudiendo identificar facilmente si existen relaciones de dependencia entre algunas variables.

# COMMAND ----------

sns.pairplot(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Visualicemos el diagrama de dispersión de tamaño del motor (<b>engine-size</b>) y precio (<b>price</b>)

# COMMAND ----------

sns.set(style="whitegrid")
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)

# COMMAND ----------

# MAGIC %md
# MAGIC <p>A medida que aumenta el tamaño del motor, aumenta el precio: esto indica una correlación directa positiva entre estas dos variables. El tamaño del motor parece ser un buen predictor de precio ya que la línea de regresión es casi una línea diagonal perfecta.</p>

# COMMAND ----------

# MAGIC %md
# MAGIC Visualicemos ahora <b>highway-mpg</b> y <b>price</b>. A medida que aumenta <b>highway-mpg</b>, el precio baja: esto indica un relación inversa/negativa entre estas dos variables. <b>highway-mpg</b> podría predecir el precio.

# COMMAND ----------

sns.set(style="whitegrid")
sns.regplot(x="highway-mpg", y="price", data=df)

# COMMAND ----------

sns.set(style="whitegrid")
sns.regplot(x="length", y="price", data=df)

# COMMAND ----------

# MAGIC %md
# MAGIC El gráfico de histograma también permite realizar un análisis sobre la variable numérica. Se utiliza el tipo de gráfico de Matplotlib hist sobre la variable age.

# COMMAND ----------

df['wheel-base'].hist(figsize = (5,5))
plt.show

# COMMAND ----------

# MAGIC %md
# MAGIC <h5>Variables categóricas</h5>
# MAGIC 
# MAGIC <p>Estas son variables que describen una ‘característica’ de una unidad de datos y se seleccionan de un pequeño grupo de categorías. Las variables categóricas pueden tener el tipo <b>objeto</b> o <b>int64</b>. Una buena forma de visualizar variables categóricas es mediante el uso de diagramas de caja.</p>

# COMMAND ----------

# MAGIC %md
# MAGIC Mostremos un solo atributo, y contemos cuantos ejemplos hay de cada categoría. En este caso el atributo <b>make</b> que es la marca del carro:

# COMMAND ----------

sns.set(style="whitegrid")
sns.catplot(x="make", kind="count", palette="ch:.25", data=df, height = 8, aspect = 3)


# COMMAND ----------

sns.set(style="whitegrid")
sns.catplot(x="body-style", y="price", hue_order="class", kind="bar", data=df)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-success alertsuccess" style="margin-top: 20px">
# MAGIC    <strong>AHORA TÚ: </strong>  
# MAGIC   <p>
# MAGIC Visualice las variables <b>stroke</b> y <b>price</b> utilizando regplot. ¿Qué tipo de relación hay entre las variables?
# MAGIC   </p>
# MAGIC </div>

# COMMAND ----------

# Escribe tu código aquí y presiona Shift+Enter para ejecutar


# COMMAND ----------

# MAGIC %md
# MAGIC Haz doble clic <b>aquí</b> para ver la solución.
# MAGIC 
# MAGIC <!-- Respuesta:
# MAGIC 
# MAGIC sns.set(style="whitegrid")
# MAGIC sns.regplot(x="stroke", y="price", data=df)
# MAGIC 
# MAGIC -->

# COMMAND ----------

# MAGIC %md
# MAGIC ##Graficar las estadísticas

# COMMAND ----------

# MAGIC %md
# MAGIC Podemos identificar la simentría de los datos utilizando gráficos de histograma:

# COMMAND ----------

#Histograma del atributo "length"
sns.distplot(df.length)

# COMMAND ----------

df['length'].hist(figsize = (6,6))
plt.show

# COMMAND ----------

sns.displot(df['length'])

# COMMAND ----------

mean = df['length'].mean()
median = df['length'].median()
mode = df['length'].mode()
skew = df['length'].skew()
kurt = df['length'].kurt()
print("La media es:", mean)
print("La mediana es:", median)
print("La moda es:", mode)
print("El sesgo es:", skew)
print("La kurtosis es:", kurt)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-success alertsuccess" style="margin-top: 20px">
# MAGIC    <strong>AHORA TÚ: </strong>  
# MAGIC   <p>
# MAGIC Seleccione una variable y visualice su histograma. ¿Qué tipo de simentría tiene?
# MAGIC   </p>
# MAGIC </div>

# COMMAND ----------

# Escribe tu código aquí y presiona Shift+Enter para ejecutar


# COMMAND ----------

# MAGIC %md
# MAGIC La dispersión de datos se puede comprobar también mediante los gráficos de tipo boxplot:

# COMMAND ----------

sns.boxplot(x=df.price)

# COMMAND ----------

# MAGIC %md
# MAGIC Representar más de una gráfico tipo boxplot permite comparar la dispersión de los datos al poder ver los resultados de forma conjunta. Veamos la relación entre <b>body-style</b> y <b>price</b>.

# COMMAND ----------

sns.boxplot(x="body-style", y="price", data=df)

# COMMAND ----------

# MAGIC %md
# MAGIC <p>Vemos que las distribuciones de precios entre las diferentes categorías de estilo de cuerpo tienen una superposición significativa, por lo que el estilo de cuerpo no sería un buen predictor del precio.</p>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-success alertsuccess" style="margin-top: 20px">
# MAGIC    <strong>AHORA TÚ: </strong>  
# MAGIC   <p>
# MAGIC Visualice las variables <b>engine-location</b> y <b>price</b> utilizando un gráfico de BoxPlot
# MAGIC   </p>
# MAGIC </div>

# COMMAND ----------

# Escribe tu código aquí y presiona Shift+Enter para ejecutar


# COMMAND ----------

# MAGIC %md
# MAGIC Haz doble clic <b>aquí</b> para ver la solución.
# MAGIC 
# MAGIC <!-- Respuesta:
# MAGIC 
# MAGIC sns.boxplot(x="engine-location", y="price", data=df)
# MAGIC 
# MAGIC -->

# COMMAND ----------

# MAGIC %md
# MAGIC ##Exportar los datos de Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC <p>
# MAGIC De la misma forma, Pandas nos permite guardar el conjunto en formato CSV con el método <code>dataframe.to_csv()</code>, puede añadir la ruta al archivo y el nombre con comillas dentro de los corchetes.
# MAGIC </p>
# MAGIC <p>
# MAGIC     Por ejemplo, si guarda el dataframe <code>df</code> como <b>automobile.csv</b> en su equipo local o en este caso dentro de Databricks, podría usar la sintaxis siguiente:
# MAGIC     </p>

# COMMAND ----------

path="/FileStore/tables/automobile.csv"
df.to_csv(path)

# COMMAND ----------

# MAGIC %md
# MAGIC Podemos leer y guardar con otros formatos y usar funciones similares a <code>pd.read_csv()</code> y <code>df.to_csv()</code> para otros formatos de datos, las funciones se muestran en la siguiente tabla:
# MAGIC 
# MAGIC | Data Formate |        Read       |            Save |
# MAGIC | ------------ | :---------------: | --------------: |
# MAGIC | csv          |  `pd.read_csv()`  |   `df.to_csv()` |
# MAGIC | json         |  `pd.read_json()` |  `df.to_json()` |
# MAGIC | excel        | `pd.read_excel()` | `df.to_excel()` |
# MAGIC | hdf          |  `pd.read_hdf()`  |   `df.to_hdf()` |
# MAGIC | sql          |  `pd.read_sql()`  |   `df.to_sql()` |
# MAGIC | ...          |        ...        |             ... |

# COMMAND ----------

# MAGIC %md
# MAGIC <p>Para descargar un archivo de Databricks, utilizamos la siguiente URL:</p>
# MAGIC <p>https://community.cloud.databricks.com/files/tables/automobile.csv?o=1396817536124706</p>
# MAGIC <p>El número que va después de o= es su número de instancia y debe buscarlo en su URL.</p>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Links de ayuda interesantes:
# MAGIC <ul>
# MAGIC     <li>Graficos Seaborn: https://seaborn.pydata.org/examples/index.html</li>
# MAGIC     <li>Plotting with categorical data: https://seaborn.pydata.org/tutorial/categorical.html</li>
# MAGIC     <li>Visualizing statistical relationships: https://seaborn.pydata.org/tutorial/relational.html </li>
# MAGIC     <li>Funciones Generales y Ayuda Pandas: https://pandas.pydata.org/pandas-docs/stable/reference/general_functions.html </li>
# MAGIC     <li>Ayuda de Pandas para función read_csv: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html?highlight=pandas%20read_csv#pandas.read_csv </li>
# MAGIC     <li>How to Perform Exploratory Data Analysis with Seaborn: https://towardsdatascience.com/how-to-perform-exploratory-data-analysis-with-seaborn-97e3413e841d</li> 
# MAGIC <ul>
