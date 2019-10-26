########################################################################
# getKindFeaturesInDataSet.py,
#
# Checks dataset given, gets data type and feature names.
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
from modulesProject.dataBase_module import ConnectDataBase
from modulesProject.dataBase_module import HandlerQuery

class checkDataSet(object):

    def __init__(self, dataSet, idDataSet):

        self.dataSet = pd.read_csv(dataSet)
        self.idDataSet = idDataSet
        self.connect = ConnectDataBase.ConnectDataBase()
        self.handler = HandlerQuery.HandlerQuery()

    #metodo que recorre toda la data e inserta los valores de las features de los set de datos...
    def insertFeaturesDB(self):

        self.getKeysInDS()#obtengo la lista de features

        for key in self.keys:#recorro la lista y obtengo el tipo por cada una
            response = self.checkKindFeature(key)
            #hacemos la insercion del feature en la bd...
            self.connect.initConnectionDB()
            query = "insert into feature values (null, '%s', '%s', %s)" % (key, response, self.idDataSet)
            self.handler.insertToTable(query, self.connect)
            self.connect.closeConnectionDB()

    #metodo que permite obtener todas las keys...
    def getKeysInDS(self):

        self.keys = []
        for element in self.dataSet:
            self.keys.append(element)


    #metodo que permite evaluar el tipo de set de datos...
    def checkKindFeature(self, feature):

        response ="CONTINUA"

        for i in range(len(self.dataSet)):

            try:
                data = float(self.dataSet[feature][i])
            except:
                response="DISCRETA"
                break

        return response
