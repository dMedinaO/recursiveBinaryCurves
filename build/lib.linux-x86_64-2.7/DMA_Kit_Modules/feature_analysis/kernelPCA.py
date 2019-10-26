########################################################################
# kernelPCA.py,
#
# Kernel PCA algorithm
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
from sklearn.decomposition import KernelPCA

#metodos de la libreria utils...
from DMA_Kit_Modules.utils import transformDataClass
from DMA_Kit_Modules.utils import transformFrequence
from DMA_Kit_Modules.utils import ScaleNormalScore
from DMA_Kit_Modules.utils import ScaleMinMax
from DMA_Kit_Modules.utils import ScaleDataSetLog
from DMA_Kit_Modules.utils import ScaleLogNormalScore
from DMA_Kit_Modules.utils import encodingFeatures

class kpca(object):

	def __init__(self, dataSet, pathResponse, optionNormalize):

		self.pathResponse = pathResponse
		self.dataSet = dataSet
		self.optionNormalize = optionNormalize
	#metodo que permite normalizar el set de datos con respecto a la opcion entregada
	def normalizeDataSet(self):
		#ahora transformamos el set de datos por si existen elementos discretos...
		#transformDataSet = transformFrequence.frequenceData(self.dataSet)
		#dataSetNewFreq = transformDataSet.dataTransform
		encoding = encodingFeatures.encodingFeatures(self.dataSet, 20)
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

	def doKPCA(self):
		#Mykernel puede ser:
		#linear
		#poly
		#rbf
		#sigmoid
		#cosine
		#precomputed
		okiedoki=""
		try:

			X_or = self.normalizeDataSet()

			high, width = X_or.shape

			transformer = KernelPCA(n_components=width, kernel='linear')
			Y = transformer.fit_transform(X_or)

				#CSV
			file = "%sKernelPCA.csv" % (self.pathResponse)

			df = pd.DataFrame(Y)
			df.to_csv(file, index=False)

			okiedoki = "OK"

		except Exception as e:
			#raise e
			okiedoki = "ERROR"
			pass
		return okiedoki
