import pandas as pd # Libreria para la manipulacion de datos
from sklearn.datasets import load_iris # Usaremos el dataset iris
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot
import matplotlib.pyplot as plt # Libreria para poder realizar la regresionç
import numpy as np

try:
	#we have all the data at hepatitis.data, so in order to work with it, we must convert it to csv. 
	#we will use pandas library to achieve this
	Columns = ["Clase","Edad","Sexo","Tratamiento con esteroides","Tratamiento antiviral","Fatiga","Malestar general","Anorexia","Hepatomegalia","Firmeza hepática","Esplenomegalia palpable","Arañas vasculares","Ascitis","Varices esofágicas","Bilirrubina sérica total","Fosfatasa alcalina sérica","Transaminasa glutámico-oxalacética sérica","Albúmina sérica","Tiempo de protrombina","Histología"]
    
	original_data_frame = pd.read_csv("hepatitis.data",names=Columns)
	#now we can start working
	original_metadata = SingleTableMetadata()
	original_metadata.detect_from_dataframe(data=original_data_frame)
	
	synthesizer = CTGANSynthesizer(original_metadata, enforce_rounding=False,epochs=500,verbose=True)

	synthesizer.fit(original_data_frame)
	
	synthetic_data = synthesizer.sample(num_rows=155)

	original_data_frame.replace('?', np.nan, inplace=True)
	synthetic_data.replace('?', np.nan, inplace=True)

	print(synthetic_data)

	quality_report = evaluate_quality(
    real_data=original_data_frame,
    synthetic_data=synthetic_data,
    metadata=original_metadata)

	fig = get_column_plot(
    real_data=original_data_frame,
    synthetic_data=synthetic_data,
    column_name='Albúmina sérica',
    metadata=original_metadata
	)
    
	fig.show()

except FileNotFoundError:
	print("Error: The file was not found")
except OSError:
	print("Error: There was an OS error")
	