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

from modulesRV import nodo
from modulesRV import callService
import graphviz as gp
import pylab
import pandas as pd
import time

class BinaryTree(object):
    def __init__(self):
        self.top = None

    # Llamada para dividir grupo de forma recursiva
    def split(self, nodoElement, dataInput_scaleDF, pathResponse, len_data, percentageMember, significanciaLevel):
        #print "Llamando a servicio -> ",dataSet.shape[0]
        callServiceObject = callService.serviceClustering(dataInput_scaleDF, pathResponse, len_data, percentageMember, significanciaLevel)
        #callService debe retornar un arreglo en donde [sePuedeDividir,dataFramegrupo1,dataFramegrupo2]
        result = callServiceObject.execProcess()
        if isinstance(result,list):
            if(result[0] == -1):
                #print "No puedo dividir: ",dataSet.shape[0]
                return nodoElement
            else:
                #print "Dividir -> ",dataSet.shape[0]
                #print "G1: ",result[1].shape[0]
                #print "G2: ",result[2].shape[0]
                #Los sleep es para generar id unicos por cada dataframe que se agrega al arbol
                nodoElement.left = nodo.Nodo(result[1])
                time.sleep(0.05)
                nodoElement.right = nodo.Nodo(result[2])
                time.sleep(0.05)
                nodoElement.left = self.split(nodoElement.left,result[1], pathResponse, len_data, percentageMember, significanciaLevel)
                time.sleep(0.05)
                nodoElement.right = self.split(nodoElement.right,result[2], pathResponse, len_data, percentageMember, significanciaLevel)
                return nodoElement
        else:
            #almacena nodoElement anterior que se pudo dividir
            dataInput_scaleDF.to_csv(pathResponse+""+str(dataInput_scaleDF.shape[0])+'_'+str(int(round(time.time() * 1000)))+'.csv')
            return nodoElement

    # Llamar funcion recursiva para dibujar arbol
    def diagramSplit(self, pathResult) :
        print "Imprimir"
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
        self.top = nodo.Nodo(data)
