import pandas as pd # Libreria para la manipulacion de datos
from sklearn.datasets import load_iris # Usaremos el dataset iris
from sdv.tabular import CTGAN # Tecnica de sintesis de datos
import matplotlib.pyplot as plt # Libreria para poder realizar la regresionç


# Cargamos el dataset de iris de scikit-learn
iris = load_iris()

# Usamos los datos del dataset como una estructura data frame gracias a panda
original_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Print del dataset original
print("Dataset original:\n")
print(original_data)

# Seleccionar las columnas a sintetizar
columns_to_synthesize = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Seleccionar el modelo de machine o deep learning a usar (por ejemplo, CTGAN)
model = CTGAN()

# Entrenar el modelo con los datos originales
model.fit(original_data[columns_to_synthesize])

# Generar nuevos datos sintéticos
synthetic_data = model.sample(len(original_data))

# Mostrar el dataset generado
print("Dataset generado:\n")
print(synthetic_data)



# Ahora vamos a realizar un analisis de regresion de la longitud y ancho de los sepalos, para ver la calidad de los datos generados

# Para generar los puntos
#plt.scatter(original_data['sepal length (cm)'], original_data['sepal width (cm)'], label='Datos originales', c ='blue')

# aplicar leyenda
#plt.xlabel('sepal length (cm)')
#plt.ylabel('sepal width (cm)')
#plt.legend()

#mostrar grafico
#plt.show()