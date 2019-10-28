########################################################################
# createCharts.py,
#
# Methods associated to graphics creations.
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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tools.plotting import parallel_coordinates

class graphicsCreator(object):

    #metodo que permite crear el grafico de torta
    def createPieChart(self, keys, values, namePicture):

        df = pd.DataFrame(values, index=keys, columns=['Groups'])
        # make the plot
        df.plot(kind='pie', subplots=True, figsize=(8, 8))
        plt.savefig(namePicture)

    #metodo que permite crear un grafico de barras
    def createBarChart(self, keys, values, xLabel, yLabel, title, namePicture):

        # Fake dataset
        y_pos = np.arange(len(keys))

        # Create bars and choose color
        plt.bar(y_pos, values, color = (0.5,0.1,0.5,0.6))

        # Add title and axis names
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)

        # Limits for the Y axis
        plt.ylim(0,100)

        # Create names
        plt.xticks(y_pos, keys)

        # Show graphic
        plt.savefig(namePicture)

    #metodo que permite crear un grafico de barras dobles comparativas
    def createBarChartCompare(self, values1, values2, label1, label2, xLabel, yLabel, title, listData, namePicture):
        plt.figure()
        listData = list(set(listData))
        # set width of bar
        barWidth = 0.25

        # Set position of bar on X axis
        r1 = np.arange(len(values1))
        r2 = [x + barWidth for x in r1]

        # Make the plot
        plt.bar(r1, values1, color='#7f6d5f', width=barWidth, edgecolor='white', label=label1)
        plt.bar(r2, values2, color='#557f2d', width=barWidth, edgecolor='white', label=label2)

        # Add xticks on the middle of the group bars
        plt.xlabel(xLabel, fontweight='bold')
        plt.ylabel(yLabel, fontweight='bold')

        plt.xticks([r + barWidth for r in range(len(values1))], listData)

        # Create legend & Show graphic
        plt.legend()
        plt.title(title)
        plt.savefig(namePicture)

    #metodo que permite crear el grafico de la matriz de confusion asociada al proceso de entrenamiento de modelos
    def createConfusionMatrixPictures(self, cm, listData, namePicture):

        normalize=False
        uniqueList = list(set(listData))
        plt.figure()
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=uniqueList, yticklabels=uniqueList,
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        plt.savefig(namePicture)

    #metodo que permite crear el grafico de scatter plot
    def createScatterPlotErrorPrediction(self, real_values, predict_values, namePicture):

        df=pd.DataFrame({'x': real_values, 'y': predict_values})

        # plot with matplotlib
        plt.plot( 'x', 'y', data=df, marker='o', linestyle='none', color='mediumvioletred')
        plt.xlabel("Real Values", fontweight='bold')
        plt.ylabel("Predict Values", fontweight='bold')
        plt.title("Scatter Plot Real v/s Predict Values")
        plt.savefig(namePicture)

    #metodo que permite crear el grafico de la visual continua
    def createScatterContinueData(self, data, namePicture, key):

        df=pd.DataFrame({'x': data})

        # plot with matplotlib
        plt.plot( 'x', data=df, marker='o', linestyle='none', color='mediumvioletred')
        plt.xlabel("Values in "+key, fontweight='bold')
        plt.title("View values for "+key)
        plt.savefig(namePicture)

    #metodo que permite crear el boxplot
    def createBoxPlot(self, dataValues, namePicture):
        plt.figure()
        sns.boxplot(data=dataValues)
        plt.title("Box Plot")
        plt.savefig(namePicture)

    #metodo que permite crear el boxplot
    def createViolinPlot(self, dataValues, namePicture):

        plt.figure()
        sns.violinplot(data=dataValues)
        plt.title("Violin Plot")
        plt.savefig(namePicture)

    #funcion que permite crear un histograma...
    def generateHistogram(self, dataSet, key, exportName, title):

        plt.figure()
        sns.set(color_codes=True)
        sns.set(style="ticks")

        #obtenemos solo los valores de interes
        dataValues = []
        for element in dataSet[key]:
            try:
                dataValues.append(float(element))
            except:
                pass
        sns_plot = sns.distplot(dataValues , color="olive", label=key, kde=False, rug=True)
        sns.plt.legend()
        sns.plt.title(title)
        sns_plot.figure.savefig(exportName)


    #funcion que permite crear un parallel coordinates
    def createParallelCoordinates(self, dataSet, key, namePicture, title):

        plt.figure()
        parallel_coordinates(dataSet, key, colormap=plt.get_cmap("Set2"))
        plt.title(title)
        plt.savefig(namePicture)

    #funcion que permite crear un scatter plot matrix
    def createScatterPlotMatrix(self, dataSet, namePicture, key):

        plt.figure()
        sns.pairplot(dataSet, kind="scatter", hue=key, palette="Set2")
        plt.savefig(namePicture)
