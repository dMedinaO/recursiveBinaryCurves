########################################################################
# execFeatureAnalysis.py,
#
# Generate instances from requests calls.
# Process options from users choices.
# Checks type of dataset, can it be clustering or element prediction.
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

from DMA_Kit_Modules.dataBase_module import ConnectDataBase
from DMA_Kit_Modules.dataBase_module import HandlerQuery
from DMA_Kit_Modules.feature_analysis import correlationValue
from DMA_Kit_Modules.feature_analysis import spatialDeformation
from DMA_Kit_Modules.feature_analysis import kernelPCA
from DMA_Kit_Modules.feature_analysis import mutualInformation
from DMA_Kit_Modules.feature_analysis import PCA_Method

class featureAnalysis(object):

    def __init__(self, dataSet, pathResponse):

        self.dataSet = dataSet
        self.pathResponse = pathResponse

    #metodo que permite la ejecucion de la correlacion de datos
    def execCorrelationData(self, optionNormalize):

        corrObject = correlationValue.correlationMatrixData(self.dataSet, self.pathResponse, optionNormalize)
        return corrObject.calculateCorrelationMatrix()

    #metodo que permite la ejecucion de la deformacion de espacio con random forest
    def excecSpatialDeformation(self, feature, kindDataSet, optionNormalize):
        spatial = spatialDeformation.spatialDeformation(self.dataSet, self.pathResponse, optionNormalize)
        return spatial.applySpatialDeformation(feature, kindDataSet)

    #metodo que permite la ejecucion de mutual information...
    def execMutualInformation(self, optionNormalize):

        mutualObject = mutualInformation.mutualInformation(self.dataSet, self.pathResponse, optionNormalize)
        return mutualObject.makeMatrix()

    #metodo que permite la ejecucion de PCA information...
    def execPCA(self, optionNormalize):

        pcaObject = PCA_Method.pca(self.dataSet, self.pathResponse, optionNormalize)
        return pcaObject.doPCA()

    #metodo que permite la ejecucion de incremental PCA...
    def execPCA_Incremental(self, optionNormalize):

        pcaObject = PCA_Method.pca(self.dataSet, self.pathResponse, optionNormalize)
        return pcaObject.incrementalPCA()

    #metodo que permite la ejecucion de kernel pca...
    def exec_kernelPCA(self, optionNormalize):

        kernelObject = kernelPCA.kpca(self.dataSet, self.pathResponse, optionNormalize)
        return kernelObject.doKPCA()
