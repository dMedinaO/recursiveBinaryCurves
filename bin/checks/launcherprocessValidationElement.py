from modulesNLM.utils import createRandomPartitions

import sys
import pandas as pd
import argparse
import os

#recibimos el conjunto de datos y el maximo de particiones, junto con el numero de veces a generar cada particion y el path
dataSet = pd.read_csv(sys.argv[1])
maxSplitter = int(sys.argv[2])
maxDistribution = int(sys.argv[3])
pathOutput = sys.argv[4]

for i in range(2, maxSplitter+1):#hacemos las iteraciones

    #creamos el directorio de la particion correspondiente
    command = "mkdir -p %s%d" % (pathOutput, i)
    os.system(command)

    for j in range(maxDistribution):
        randomDist = createRandomPartitions.createRandomDistribution(dataSet, i)#entregamos el set de datos y el numero de divisiones
        randomDist.createRandomValuesDistribution()
        break
    break
