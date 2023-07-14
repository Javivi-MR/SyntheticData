import pandas as pd # Libreria para la manipulacion de datos
from sklearn.datasets import load_iris # Usaremos el dataset iris
from sdv.single_table import CTGANSynthesizer # Importamos el sintetizador
from sdv.metadata import SingleTableMetadata # Importamos el modulo para crear la tabla de metadatos
from sdv.evaluation.single_table import evaluate_quality # importamos el modulo para la evaluacion de los datos
from sdv.evaluation.single_table import get_column_plot 
import matplotlib.pyplot as plt # Libreria para poder realizar la regresionç
import numpy as np

try:
	#Los datos estan almacenados en adult_reduced.data, asi que para poder trabajar con estos, debemos usar un dataframe.
	#Para ello usaremos la libreria pandas
	Columns = ["Edad","Clase de trabajo","Peso final","Tipo de educación", "Años de educación","Estado civil", "Ocupacion actual", "Relacion familiar", "Raza", "Género", "Ganancia de capital", "Perdida de capital", "Horas trabajadas por semana", "País de origen", "Ingreso"]
    
    #Creamos un data frame con los datos originales
	original_data_frame = pd.read_csv("adult_reduced.data",names=Columns)
	#Creamos la tabla de metadatos
	original_metadata = SingleTableMetadata()
	original_metadata.detect_from_dataframe(data=original_data_frame)
	#Configuramos el modelo, en este caso usamos CTGAN
	synthesizer = CTGANSynthesizer(original_metadata, enforce_rounding=False,epochs=500,verbose=True)
    #Ponemos el modelo a entrenar
	synthesizer.fit(original_data_frame)
	#Una vez entrenado, cogemos una muestra de 155 filas
	synthetic_data = synthesizer.sample(num_rows=155)

	print(synthetic_data)

    #Creamos el informe de evalucion
	quality_report = evaluate_quality(
    real_data=original_data_frame,
    synthetic_data=synthetic_data,
    metadata=original_metadata)

    #Creamos el grafico de comparacion
	fig = get_column_plot(
    real_data=original_data_frame,
    synthetic_data=synthetic_data,
    column_name='Ingreso',
    metadata=original_metadata)
    #mostramos el grafico
	fig.show()

except FileNotFoundError:
	print("Error: The file was not found")
except OSError:
	print("Error: There was an OS error")
	