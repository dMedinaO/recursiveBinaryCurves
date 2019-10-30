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

import pandas as pd
import os

class evaluatedClusteringProcess(object):

    def __init__(self, pathResult, header):

        self.pathResult = pathResult
        self.header = header

        self.getNameFilesInResponse()

    #metodo que permite poder obtener
    def getNameFilesInResponse(self):

        listFiles = os.listdir(self.pathResult)

        #obtenemos solo los archivos en formato *.csv
        self.listCluster = []
        for element in listFiles:
            if ".csv" in element:
                self.listCluster.append(element)

    #metodo que permite formar el unico conjunto de datos para hacer la
