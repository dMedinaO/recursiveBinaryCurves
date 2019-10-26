import sys
import os
import pandas as pd
import json

from modulesNLM.utils import checkFeatureNoInformative

pathPartitions = sys.argv[1]
performance = sys.argv[2]
kindDataSet = int(sys.argv[3])
numberPartitions = int(sys.argv[4])

dictResponseFull = []
ListAccuracy = []
ListPrecision = []
ListRecall = []
ListF1 = []
ListMembers = []

for i in range(1, numberPartitions+1):


    dataSet = pathPartitions+"p"+str(i)+"/p"+str(i)+".csv"
    dataSetFrame = pd.read_csv(dataSet)
    membersLenght = len(dataSetFrame)

    checkFeatures = checkFeatureNoInformative.checkNonInformativeFeatures(dataSetFrame, dataSet)
    checkFeatures.evalNonInformative()

    pathOutput = pathPartitions+"p"+str(i)+"/"

    if kindDataSet ==1:#clasificacion
        command = "python /home/dmedina/Escritorio/MyProjects/UChileProjects/nonlinearModels/bin/launcherScanClassification.py -d %s -p %s -m %s" % (dataSet, pathOutput, performance)

    else:
        command = "python /home/dmedina/Escritorio/MyProjects/UChileProjects/nonlinearModels/bin/launcherScanPrediction.py -d %s -p %s -m %s" % (dataSet, pathOutput, performance)

    os.system(command)

    jsonResponse = pathOutput+"summaryProcess.json"

    #adicionamos la informacion del modelo y la medida de desempeno
    with open(jsonResponse) as json_file:
        dataJSON = json.load(json_file)
        paramsResponse = dataJSON['modelSelecetd']

    accuracyMax =0
    model = {}
    measures = {}
    algorithm = 0
    for i in range(len(paramsResponse)):

        if paramsResponse[i]['Accuracy'] >=accuracyMax:

            #obtenemos las medidas
            measures.update({'Accuracy': paramsResponse[i]['Accuracy']})
            measures.update({'Precision': paramsResponse[i]['Precision']})
            measures.update({'Recall': paramsResponse[i]['Recall']})
            measures.update({'F1': paramsResponse[i]['F1']})

            #obtemos el algoritmo
            algorithm = paramsResponse[i]['algorithm']

            #obtenemos los parametros
            for key in paramsResponse[i]['params']:
                model.update({key:paramsResponse[i]['params'][key]})
    modelDefinition = {}
    modelDefinition.update({'members':membersLenght})
    modelDefinition.update({'algorithm':algorithm})
    modelDefinition.update({'measures':measures})
    modelDefinition.update({'model':model})
    modelDefinition.update({'featuresRemove':checkFeatures.listRemoveFeatures})
    dictResponseFull.append(modelDefinition)

    #agregamos las medidas a la lista de desempeno para procesar la medida general
    ListAccuracy.append(modelDefinition['measures']['Accuracy'])
    ListPrecision.append(modelDefinition['measures']['Precision'])
    ListRecall.append(modelDefinition['measures']['Recall'])
    ListF1.append(modelDefinition['measures']['F1'])
    ListMembers.append(modelDefinition['members'])

#hacemos la medida de desempeno ponderada
accuracyPond = 0
precisionPond = 0
recallPond = 0
f1Pond = 0

for i in range(len(ListAccuracy)):
    aporteA = ListAccuracy[i]
    aporteP = ListPrecision[i]
    aporteF = ListF1[i]
    aporteR = ListRecall[i]

    accuracyPond+=aporteA
    precisionPond+=aporteP
    recallPond+=aporteR
    f1Pond+=aporteF

accuracyPond= accuracyPond/numberPartitions
precisionPond= precisionPond/numberPartitions
recallPond= recallPond/numberPartitions
f1Pond= f1Pond/numberPartitions

dictPonderated = {'AccuracyPond':accuracyPond, 'PrecisionPond':precisionPond, 'RecallPond':recallPond, 'F1Pond':f1Pond}

print dictPonderated
dictResponseFull.append({'MeasureFull':dictPonderated})

nameFileExport = pathPartitions+"SummaryAllModels.json"
with open(nameFileExport, 'w') as fp:
    json.dump(dictResponseFull, fp)
