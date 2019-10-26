########################################################################
# LauncherPipeLine.py,
#
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

import pandas as pd
import argparse
import os
from modulesNLM.checks_module import checkNonLinearRegression, checkNonLinearClass

#inputs: dataset, path output, tipoDataset, FeatureResponse, umbralResponse, percetange minimun for group division, umbral unbalance class

#add params to list
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataSet", help="Full path and name to acces dataSet input process", required=True)
parser.add_argument("-p", "--pathResult", help="Full path for save results", required=True)
parser.add_argument("-m", "--performance", help="Performance selected model", required=True)
parser.add_argument("-k", "--kindDataSet",type=int, help="Kind of data set: 1. for classifiers 2. for regression models", required=True)
parser.add_argument("-f", "--feature", help="Name of feature response in dataset", required=True)
parser.add_argument("-u", "--threshold",type=float, help="Threshold for umbral response acceptance", required=True)
parser.add_argument("-x", "--percentage", type=float, help="Minimun percentage of members in group", required=True)
parser.add_argument("-a", "--proportionClass", type=float, help="Minimun porcentage acceptance for unbalance class", required=True)

args = parser.parse_args()

#primer paso: Evaluar linealidad del conjunto de datos
print "Check non linear model"

#get values of params
dataSet = pd.read_csv(args.dataSet)
featureResponse = args.feature
threshold = args.threshold

if args.kindDataSet == 1:#class
    checkMethod = checkNonLinearClass.checkNonLinearClass(dataSet, featureResponse, threshold)
    checkMethod.prepareDataSet()#preparamos el conjunto de datos
    response = checkMethod.applyTraining()#aplicamos la regresion lineal

else:
    checkMethod = checkNonLinearRegression.checkNonLinearRegression(dataSet, featureResponse, threshold)
    checkMethod.prepareDataSet()#preparamos el conjunto de datos
    response = checkMethod.applyLinearRegression()#aplicamos la regresion lineal

if response == 0:
    print "dataset is nonlinear"

    #create directory for exploratory step
    command = "mkdir -p %sExploratoryTraining" % (args.pathResult)
    pathExportTraining = "%sExploratoryTraining/" % (args.pathResult)
    os.system(command)

    command = ""
    #obtenemos los mejores modelos con respoecto a la etapa exploratoria
    if args.kindDataSet == 1:#class
        command = "python /home/dmedina/Escritorio/MyProjects/UChileProjects/nonlinearModels/bin/launcherScanClassification.py -d %s -p %s -m %s" % (args.dataSet, pathExportTraining, args.performance)
    else:
        command = "python /home/dmedina/Escritorio/MyProjects/UChileProjects/nonlinearModels/bin/launcherScanPrediction.py -d %s -p %s -m %s" % (args.dataSet, pathExportTraining, args.performance)

    print command
    os.system(command)
else:
    print "dataset is linear"
