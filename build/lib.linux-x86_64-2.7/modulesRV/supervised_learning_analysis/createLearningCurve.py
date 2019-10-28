########################################################################
# createLearningCurve.py,
#
# Create Roc curve of given model about number of validations generated.
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

import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

class curveLearning(object):

    def __init__(self, dataSet, target, modelData, cv_values, path):

        self.dataSet = dataSet
        self.target = target
        self.modelData = modelData
        self.cv_values = 10
        self.path = path
        self.pathResponse = self.path+"curveLearning.svg"

    def createLearningCurve(self):

        X = np.array(self.dataSet)
        y = np.array(self.target)

        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = ShuffleSplit(n_splits=100, test_size=self.cv_values/100, random_state=0)
        self.plot_learning_curve(self.modelData, 'Learning Curve', X, y, self.cv_values, ylim=(0.01, 1.01), n_jobs=4)

        plt.savefig(self.pathResponse)

    #funcion que permite crear la curva ROC y exportar la imagen segun corresponda
    def plot_learning_curve(self, estimator, title, X, y, cv, ylim=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt
