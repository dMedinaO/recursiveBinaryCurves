########################################################################
# responseResults.py,
#
# Response Result
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
import os
from random import uniform

class responseProcess(object):

    #metodo que permite crear un directorio donde se almacene la informacion
    def createDirResult(self, path):

        if path[-1] != "/":
            path = path+"/"

        randomDir = str(uniform(0,10000)).replace(".","")
        nameDir = "%s%s/" % (path, randomDir)
        command = "mkdir -p %s" % nameDir
        os.system(command)

        return nameDir

    #metodo que permite generar un archivo json
    def createJSONFile(self, dictResponse, nameFile):

        with open (nameFile, 'w') as fp:
            json.dump(dictResponse, fp)

    #metodo que permite validar si el data set existe
    def validateDataSetExist(self, dataSet):

        if os.path.isfile(dataSet):
            return 0#exist!
        else:
            return 1#not exist!

    #metodo que permite validar si existe el directorio output
    def validatePath(self, pathData):

        if os.path.exists(pathData):
            return 0#exist!
        else:
            return 1#not exist!
