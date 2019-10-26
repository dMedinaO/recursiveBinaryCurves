## NOTE: esto debe realizarse antes de la validacion de las particiones y eliminacion de los elementos no informativos
import sys
import pandas as pd
import os

#recibimos los parametros
pathPartitions = sys.argv[1]
numberPartitions = int(sys.argv[2])

matrixData = []
header = []

#formamos el conjunto de datos, para ello hacemos la lectura de los elementos existentes en cada particion
for i in range(1, numberPartitions+1):

    #hacemos la lectura de los datos
    nameDoc = pathPartitions+"p"+str(i)+"/"+"p"+str(i)+".csv"
    dataSet = pd.read_csv(nameDoc)

    if i == 1:#obtenemos el header
        for element in dataSet:
            header.append(element)
        header.append("partitions")

    #obtenemos la informacion de los elementos en las particiones
    for j in range(len(dataSet)):
        row = []
        for element in dataSet:
            row.append(dataSet[element][j])
        classData = "C"+str(i)
        row.append(classData)#agregamos la particion
        matrixData.append(row)

#generamos el dataframe
dataFrame = pd.DataFrame(matrixData, columns=header)
nameDocElement = pathPartitions+"dataAddPartitions.csv"
dataFrame.to_csv(nameDocElement, index=False)

#creamos el directorio para almacenar los resultados
command = "mkdir -p %sexplore" % pathPartitions
namePathResponse = pathPartitions+"explore/"

os.system(command)

#hacemos el entrenamiento de los modelos...
command = "python LauncherExploratoryClassifierModels.py -d %s -p %s -m %s -r %s -t %s -k %s" % (nameDocElement, namePathResponse, 'Accuracy', 'partitions', '0.7', '10')
print command
#os.system(command)
