########################################################################
# NuSVR.py,
#
# Script that allows to generate the installation of the project and the
# python modules in the directory of the same, in order to be able to be
# recognized from any python call and be indexed within the library itself.
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

#modules import
from sklearn.svm import NuSVR

class NuSVRModel(object):

    #building
    def __init__(self,dataset, response, kernel, degree, gamma, nu):

        #init attributes values...
        self.dataset=dataset
        self.response=response
        self.kernel=kernel
        self.degree = degree
        self.gamma = gamma
        self.nu = nu

    #instance training...
    def trainingMethod(self):

        self.model=NuSVR(kernel=self.kernel, degree=self.degree, gamma=self.gamma, nu=self.nu)
        self.SVRAlgorithm =self.model.fit(self.dataset,self.response)
        self.predicctions = self.SVRAlgorithm.predict(self.dataset)
        self.r_score = self.SVRAlgorithm.score(self.dataset, self.response)
