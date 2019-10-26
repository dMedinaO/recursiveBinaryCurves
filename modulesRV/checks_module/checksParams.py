########################################################################
# cehcksParams.py,
#
# Checks correct args given
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

class checkParams(object):

    #metodo que permite revisar los parametros del tipo cluster
    def checkParamsCluster(self, dataParams, algorithm):

        print algorithm
        if algorithm ==1 or algorithm == 2: #Birch o kmeans...
            try:
                data = int(dataParams)
                if data <2:
                    return "Number K is not valid, please input a k value higher 2"
                else:
                    return "OK"
            except:
                return "Number K is not valid, please input a k value higher 2"
                pass
        else:
            if algorithm == 3:#Agglomerative
                data = dataParams.split("-")
                if len(data) != 3:
                    return "Please check input params"
                else:
                    if data[0] in ['ward', 'complete', 'average', 'single']:
                        if data[1] in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']:
                            try:
                                dataValues = int(data[2])
                                if dataValues >=2:
                                    return "OK"
                                else:
                                    return "Number K is not valid"
                            except:
                                return "Please add a valid number for K"
                        else:
                            return "The value for Affinity is not correct, please select some of the list: [euclidean, l1, l2, manhattan, cosine, precomputed]"
                    else:
                        return "The value for linkage is not correct, please select some of the list: [ward, complete, average, single]"
            else:
                return "OK"
