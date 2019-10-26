import sys
import pandas as pd

#funcion que permite obtener el ambiente bajo el cual se encuentra la mutacion
def getEnviromentMutation(pos, pdbCode, dictSequences, codeResidues):

    arrayEnviroment = []#sera el arreglo de las codificaciones de los residuos
    residuesNear = []#sera el arreglo de los residuos

    for i in range(20):
        arrayEnviroment.append(0)
        residuesNear.append(0)

    sequence = dictSequences[pdbCode]

    #evaluamos los limites superiores a trabajar
    posMax = pos+10
    posMin = pos-10

    #completamos la parte mayor
    index=10
    if posMax<=len(sequence):
        for i in range(pos, posMax):
            residuesNear[index] = sequence[i]
            index+=1
    else:#me faltan, debo seguir al principio
        dif = posMax - len(sequence)
        for i in range(pos, len(sequence)):
            residuesNear[index] = sequence[i]
            index+=1
        #ahora trabajo con la diferencia
        for i in range(dif):
            residuesNear[index] = sequence[i]
            index+=1

    #completamos la parte inferior
    index=0
    if posMin>0:
        for i in range(posMin, pos):
            residuesNear[index] = sequence[i]
            index+=1

    else:#me sobran, debo seguir por los ultimos
        dif = 10-pos
        for i in range (pos):
            residuesNear[index] = sequence[i]
            index+=1

        for i in range(len(sequence)-dif, len(sequence)):
            residuesNear[index] = sequence[i]
            index+=1

    for i in range(len(codeResidues)):
        if codeResidues[i] in residuesNear:
            arrayEnviroment[i] =1
    return arrayEnviroment

#con respecto al orden de los residuos formamos la matriz de elementos
codeResidues = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
codeResidues.sort()

print codeResidues

listPDB = ['1A22_SDM','1CH0_SDM','1DKT_SDM','1FKJ_SDM','1FTG_SDM','1PPF_SDM','1RX4_SDM','1WQ5_SDM','2AFG_SDM','3SGB_SDM','5AUZ_SDM']

#obtenemos las secuencias y hacemos el almacenamiento de la informacion
pathSequence = sys.argv[1]
dataSet = pd.read_csv(sys.argv[2])

sequencesData = {}

for codePDB in listPDB:
    codePDB = codePDB.split("_")[0]
    codePDB = codePDB.lower()

    nameDoc = pathSequence+codePDB+".fasta.txt"

    fileOpen = open(nameDoc, 'r')
    line = fileOpen.readline().replace("\n", "")

    sequencesData.update({codePDB:line})
    fileOpen.close()

matrixEnviroment = []#almacenara el ambiente de cada mutacion es una matriz de nx20 donde n son la cantidad de ejemplos y 20 los 20 residuos ordenados alfabeticamente

#recorremos la data del dataset
for i in range(len(dataSet)):

    #tomamos la informacion
    pos = dataSet['Pos'][i]
    pdbCode = dataSet['PDBFile'][i].split(".")[0].lower()
    if pdbCode == "1cho":
        pdbCode = "1ch0"
    if pdbCode == "5azu":
        pdbCode = "5auz"

    matrixEnviroment.append(getEnviromentMutation(pos, pdbCode, sequencesData, codeResidues))

#invertimos la matriz y la agergamos al dataset
matrixEnviromentTranspose = []

for i in range(20):
    row = []
    for j in range(len(matrixEnviroment)):
        row.append(matrixEnviroment[j][i])
    matrixEnviromentTranspose.append(row)

index=0
for residue in codeResidues:
    dataSet[residue] = matrixEnviromentTranspose[index]
    index+=1

dataSet.to_csv(sys.argv[2], index=False)
