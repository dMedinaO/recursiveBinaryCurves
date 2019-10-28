########################################################################
# Copyright (C) 2019  David Medina Ortiz, david.medina@cebib.cl
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
########################################################################

import sys
import pandas as pd

#preprocessing modules
from modulesRV.checks_module import checkDataSet
from modulesRV.utils import preprocesingDataSet

#clustering modules
from modulesRV.clustering_analysis import makeRecursiveExec

print "GET PARAMS FROM COMMAND LINE"
#recibimos la data
dataInput = pd.read_csv(sys.argv[1])
pathResponse = sys.argv[2]
optionScale = int(sys.argv[3])
percentageMember = float(sys.argv[4])
significanciaLevel = float(sys.argv[5])

#obtenemos el header...
header = []
for key in dataInput:
    header.append(key)

print "PREPROCESSING DATA SET"
#preprocesamiento del dataSet
checkValues = checkDataSet.checkDataSet(dataInput)
checkValues.checkNullValues()
dataInput_removeNull = checkValues.dataSet

#estandarizacion
preprocessing = preprocesingDataSet.preprocessingUtil(dataInput_removeNull, optionScale)
preprocessing.applyScale()
dataInput_scale = preprocessing.dataSet
dataInput_scaleDF =pd.DataFrame(dataInput_scale, columns=header)

print "CLUSTERING PROCESS"

tree = makeRecursiveExec.BinaryTree()
#Nodo raiz con la informacion del dataSet inicial
initialSize = dataInput_scaleDF.shape[0]
tree.insert(dataInput_scaleDF)
tree.split(tree.top,dataInput_scaleDF, pathResponse, initialSize, percentageMember, significanciaLevel)
tree.diagramSplit(pathResponse)
