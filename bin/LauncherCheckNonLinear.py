########################################################################
# callService.py,
#
# Execute the clustering service using job process, for wich a service call in background need to be done.
# All clustering process will be addressed with their own parameter and answers.
#
# Receives a normalized dataset, job id and user id. The data will be used to save information in user's personal path.
#
# Returns information as an csv file, calinski medition and sihouette through two histogram. Finally a json file with the process resume.
#
# NOTE: in the clustering service all methods are tested without the user confirmation. Thus this service implementation is easier than the supervised learning service
#
# python modules in the directory of the same, in order to be able to be
# recognized from any python call and be indexed within the library itself.
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

from modulesNLM.checks_module import checkNonLinearRegression, checkNonLinearClass
import sys
import pandas as pd

#atributos
dataSet = pd.read_csv(sys.argv[1])

tipo = int(sys.argv[2])
featureResponse = sys.argv[3]
threshold = float(sys.argv[4])

response = -1

if tipo == 1:#class
    checkMethod = checkNonLinearClass.checkNonLinearClass(dataSet, featureResponse, threshold)
    checkMethod.prepareDataSet()#preparamos el conjunto de datos
    response = checkMethod.applyTraining()#aplicamos la regresion lineal

else:
    checkMethod = checkNonLinearRegression.checkNonLinearRegression(dataSet, featureResponse, threshold)
    checkMethod.prepareDataSet()#preparamos el conjunto de datos
    response = checkMethod.applyLinearRegression()#aplicamos la regresion lineal

if response == 0:
    print "Dataset is nonlinear"
elif response == 1:
    print "Dataset is linear "
else:
    print "Error during exec process, please contac us at developer of library"
