########################################################################
# processClustering.py,
#
# Aplicate clustering methods for a given dataset
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

from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN, Birch
from sklearn import metrics
from sklearn.metrics import pairwise_distances

class aplicateClustering(object):

    def __init__(self, dataSet):
        self.dataSet = dataSet

    #metodo que permite aplicar k-means, genera diversos set de datos con respecto a las divisiones que se emplean...
    def aplicateKMeans(self, numberK):

        try:
            self.model = KMeans(n_clusters=numberK, random_state=1).fit(self.dataSet)
            self.labels = self.model.labels_
            return 0
        except:
            pass
            return 1

    #metodo que permite aplicar birch clustering
    def aplicateBirch(self, numberK):

        try:
            self.model = Birch(threshold=0.2, branching_factor=50, n_clusters=numberK, compute_labels=True, copy=True).fit(self.dataSet)
            self.labels = self.model.labels_
            return 0
        except:
            pass
            return 1

    #metodo que permite aplicar cluster jerarquico
    def aplicateAlgomerativeClustering(self, linkage, affinity, numberK):

        try:
            self.model = AgglomerativeClustering(n_clusters=numberK, affinity=affinity, memory=None, connectivity=None, compute_full_tree='auto', linkage=linkage).fit(self.dataSet)
            self.labels = self.model.labels_
            return 0
        except:
            pass
            return 1

    #metodo que permite aplicar AffinityPropagation, con diversos parametros...
    def aplicateAffinityPropagation(self):

        try:
            self.model = AffinityPropagation().fit(self.dataSet)
            self.labels = self.model.labels_
            return 0
        except:
            pass
            return 1

    #metodo que permite aplicar DBSCAN
    def aplicateDBSCAN(self):

        try:
            self.model = DBSCAN(eps=0.3, min_samples=10).fit(self.dataSet)
            self.labels = self.model.labels_
            return 0
        except:
            pass
            return 1

    #metodo que permite aplicar MeanShift clustering...
    def aplicateMeanShift(self):

        try:
            bandwidth = estimate_bandwidth(self.dataSet, quantile=0.2)
            self.model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            self.model = self.model.fit(self.dataSet)
            self.labels = self.model.labels_
            return 0
        except:
            return 1
