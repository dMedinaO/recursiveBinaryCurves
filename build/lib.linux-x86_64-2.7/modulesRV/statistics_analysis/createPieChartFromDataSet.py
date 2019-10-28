########################################################################
# createPieChartFromDataSet.py,
#
# Pie chart maker.
# Receives a dataset,a column key of it, and the filename for the output
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

import json

class pieChart(object):

    #definimos los atributos para la clase
    def __init__(self, dataSet, key):

        self.dataSet = dataSet
        self.key = key

    #creamos la respuesta para el manejo del set de datos...
    def createExport(self):

        dataInformation = self.dataSet[self.key]

        #obtenemos las key y generamos los contadores
        keys = list(set(dataInformation))
        counts = []

        for element in keys:
            cont=0
            for data in dataInformation:
                if data == element:
                    cont+=1
            counts.append(cont)

        dicResponse = {}
        dicResponse.update({"keys":keys})
        dicResponse.update({"values":counts})
        print json.dumps(dicResponse)
