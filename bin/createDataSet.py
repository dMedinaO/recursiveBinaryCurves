import pandas as pd
import sys

dataFrame = pd.read_csv(sys.argv[1])

#creamos los set de datos independientes
dataGlucose = pd.DataFrame()
dataInsuline = pd.DataFrame()

#formamos los archivos de salida de glucosa
dataGlucose['G0'] = dataFrame['G0']
dataGlucose['G30'] = dataFrame['G30']
dataGlucose['G60'] = dataFrame['G60']
dataGlucose['G90'] = dataFrame['G90']
dataGlucose['G120'] = dataFrame['G120']

#formamos los archivos de salida de glucosa
dataInsuline['I0'] = dataFrame['I0']
dataInsuline['I30'] = dataFrame['I30']
dataInsuline['I60'] = dataFrame['I60']
dataInsuline['I90'] = dataFrame['I90']
dataInsuline['I120'] = dataFrame['I120']

#exportamos los set de datos
dataGlucose.to_csv("glucose.csv", index=False)
dataInsuline.to_csv("insuline.csv", index=False)
