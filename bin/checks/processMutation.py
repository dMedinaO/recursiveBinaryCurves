import sys
import pandas as pd

dataSet = pd.read_csv(sys.argv[1])
mutations = dataSet['Mutation']

residuesW = []
residuesM = []
posResidue = []

for mutation in mutations:
    mutation = mutation.replace(' ', "")

    residuesW.append(mutation[0])
    posResidue.append(mutation[1:-1])
    residuesM.append(mutation[-1])

#adicionamos las columnas al dataset
dataSet['wildResidue'] = residuesW
dataSet['mutationResidue'] = residuesM
dataSet['posResidue'] = posResidue

#removemos la columna de las mutaciones
del dataSet['Mutation']
dataSet.to_csv(sys.argv[1], index=False)
