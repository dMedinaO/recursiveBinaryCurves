import pandas as pd
from sklearn.metrics import accuracy_score

docReader = pd.read_csv('responseP1.csv')

realValues = docReader['Real']
predictValues = docReader['Prediccion']

#calculamos la Accuracy
response = accuracy_score(realValues, predictValues)
print response
