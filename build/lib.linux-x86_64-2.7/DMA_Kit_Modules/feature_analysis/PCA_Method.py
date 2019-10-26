# coding=utf-8
########################################################################
# PCA_Method.py,
#
# Execute PCA feature analysis.
#Receives dataset without classes labels
#
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

import numpy as np
from scipy import stats
import pandas as pd
from sklearn.decomposition import IncrementalPCA

#metodos de la libreria utils...
from DMA_Kit_Modules.utils import transformDataClass
from DMA_Kit_Modules.utils import transformFrequence
from DMA_Kit_Modules.utils import ScaleNormalScore
from DMA_Kit_Modules.utils import ScaleMinMax
from DMA_Kit_Modules.utils import ScaleDataSetLog
from DMA_Kit_Modules.utils import ScaleLogNormalScore

from DMA_Kit_Modules.graphic import createCharts
from DMA_Kit_Modules.utils import encodingFeatures


class pca(object):

	def __init__(self, dataset, pathResponse, optionNormalize):
		self.pathResponse = pathResponse
		self.dataset = dataset
		self.optionNormalize = optionNormalize

	#metodo que permite normalizar el set de datos con respecto a la opcion entregada
	def normalizeDataSet(self):
		#ahora transformamos el set de datos por si existen elementos discretos...
		#transformDataSet = transformFrequence.frequenceData(self.dataset)
		#dataSetNewFreq = transformDataSet.dataTransform
		encoding = encodingFeatures.encodingFeatures(self.dataset, 20)
		encoding.evaluEncoderKind()
		dataSetNewFreq = encoding.dataSet
		dataSetNorm = ""

		#ahora aplicamos el procesamiento segun lo expuesto
		if self.optionNormalize == 1:#normal scale
			applyNormal = ScaleNormalScore.applyNormalScale(dataSetNewFreq)
			dataSetNorm = applyNormal.dataTransform

		if self.optionNormalize == 2:#min max scaler
			applyMinMax = ScaleMinMax.applyMinMaxScaler(dataSetNewFreq)
			dataSetNorm = applyMinMax.dataTransform

		if self.optionNormalize == 3:#log scale
			applyLog = ScaleDataSetLog.applyLogScale(dataSetNewFreq)
			dataSetNorm = applyLog.dataTransform

		if self.optionNormalize == 4:#log normal scale

			applyLogNormal = ScaleLogNormalScore.applyLogNormalScale(dataSetNewFreq)
			dataSetNorm = applyLogNormal.dataTransform

		return dataSetNorm

	def doPCA(self):

		okidokie = ""
		try:
			X_or = self.normalizeDataSet()

			#PCA
			X = stats.zscore(X_or, axis=0)
			high, width = X.shape
			V = np.cov(X.T)
			values, vectors = np.linalg.eig(V)

			eig_pairs = [(abs(values[i]), vectors[:,i]) for i in range(len(values))]
			eig_pairs.sort()				#ordena menor a mayor
			eig_pairs.reverse()				#idem

			suma = sum(values)			#suma los valores eigen
			pct = [(i*100)/suma for i in sorted(values, reverse=True)]			#saca los pct de  peso de cada caracteristica

			aux= 0
			W = np.empty((width,0))				# se crea 1 matriz vacia
			P = np.empty((0,2))
			for i in (pct):
				W = np.hstack((W, eig_pairs[aux][1].reshape(width,1)))
				P = np.vstack((P,[aux+1, i]))
				aux+=1

			Y = X.dot(W)
					#############################
					#Archivos y cosas
			file = "%sTransformedPCA.csv" % (self.pathResponse)
			filePCT = "%sPCTPCA.csv" % (self.pathResponse)

			df = pd.DataFrame(Y)
			df.to_csv(file, index=False)

			dfPct= pd.DataFrame(P, columns=["Component", "Relevance"])
			dfPct.to_csv(filePCT, index=False)

			#generamos el grafico de las relevancias
			keys = dfPct['Component']
			values = dfPct['Relevance']
			namePicture = self.pathResponse+"RelevanceRanking_PCA.png"

			#instanciamos un objeto del tipo graph
			graph = createCharts.graphicsCreator()
			graph.createBarChart(keys, values, 'Component', 'Relevance (%)', 'Ranking Relevance Components', namePicture)

			okidokie = "OK"
		except Exception as e:
			#raise e
			okidokie = "ERROR"
			pass
		return okidokie

	def incrementalPCA(self):

		okidokie = ""
		try:
			X_or = self.normalizeDataSet()

			high, width = X_or.shape
			trans = IncrementalPCA(n_components=width)
			Y = trans.fit_transform(X_or)

			#CSV
			file = "%sIncrementalPCA.csv" % (self.pathResponse)
			df = pd.DataFrame(Y)
			df.to_csv(file, index=False)

			#varianza explicada
			explaindVariance = trans.explained_variance_ratio_

			matrix = []
			index=1
			for element in explaindVariance:
				component = "PCA "+str(index)
				row = [component, element*100]
				matrix.append(row)
				index+=1

			fileExport = "%svarianzaExplained.csv" % (self.pathResponse)
			dfVar = pd.DataFrame(matrix, columns=["Component", "Variance"])
			dfVar.to_csv(fileExport, index=False)

			#generamos el grafico de las relevancias
			keys = dfVar['Component']
			values = dfVar['Variance']
			namePicture = self.pathResponse+"RelevanceRanking_IPCA.png"

			#instanciamos un objeto del tipo graph
			graph = createCharts.graphicsCreator()
			graph.createBarChart(keys, values, 'Component', 'Relevance (%)', 'Ranking Relevance Components', namePicture)


			okidokie = "OK"

		except Exception as e:
			#raise e
			okidokie = "ERROR"
			pass
		return okidokie
