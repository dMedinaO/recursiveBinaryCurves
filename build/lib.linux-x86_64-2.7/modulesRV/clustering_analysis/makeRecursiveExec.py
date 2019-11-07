########################################################################
# Copyright (C) 2019  Gonzalo Munoz Rojas
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

from modulesRV.clustering_analysis import nodoClass
from modulesRV.clustering_analysis import callService
import graphviz as gp
import pylab
import pandas as pd
import time

class BinaryTree(object):
    def __init__(self):
        self.top = None

    # Llamada para dividir grupo de forma recursiva
    def split(self, nodo, dataSet, pathResponse, sizeEval, percentage, significancia):
        #print "Llamando a servicio -> ",dataSet.shape[0]
        callServiceObject = callService.serviceClustering(dataSet, pathResponse, sizeEval, percentage, significancia)
        #callService debe retornar un arreglo en donde [sePuedeDividir,dataFramegrupo1,dataFramegrupo2]
        result = callServiceObject.execProcess()
        if isinstance(result,list):
            if(result[0] == -1):
                #print "No puedo dividir: ",dataSet.shape[0]
                dataSet.to_csv(pathResponse+""+str(dataSet.shape[0])+'_'+str(int(round(time.time() * 1000)))+'.csv', index=False)
                return nodo
            else:
                #Los sleep es para generar id unicos por cada dataframe que se agregaal arbol
                nodo.left = nodoClass.Nodo(result[1])
                time.sleep(0.05)
                nodo.right = nodoClass.Nodo(result[2])
                time.sleep(0.05)
                nodo.left = self.split(nodo.left,result[1], pathResponse, sizeEval, percentage, significancia)
                time.sleep(0.05)
                nodo.right = self.split(nodo.right,result[2], pathResponse, sizeEval, percentage, significancia)
                return nodo
        else:
            #almacena nodo anterior que se pudo dividir
            dataSet.to_csv(pathResponse+""+str(dataSet.shape[0])+'_'+str(int(round(time.time() * 1000)))+'.csv', index=False)
            return nodo

    # Llamar funcion recursiva para dibujar arbol
    def diagramSplit(self, pathResult) :
        tree = gp.Graph(format='png')
        if(self.top != None):
            self.draw(self.top,tree);
        # formatear pathResult quitando ultimo slash
        pathResult=pathResult[0:(len(pathResult)-1)]
        filename = tree.render(filename='tree',directory=pathResult)

    # Renderiza arbol de clustering
    def draw(self,data,tree):
        if(data.left != None):
            tree.edge(str(data.id),str(data.left.id));
            self.draw(data.left,tree)
        tree.node(str(data.id),str(data.data.shape[0]))
        if(data.right != None):
            tree.edge(str(data.id),str(data.right.id));
            self.draw(data.right,tree)

    # Insercion para el nodo raiz
    def insert(self, data):
        self.top = nodoClass.Nodo(data)
