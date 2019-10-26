########################################################################
# getResultsFromTest.py,
#
# Eval results of different test associated to data values.
#
# Copyright (C) 2019  David Medina Ortiz david.medina@cebib.cl
#
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

import numpy as np
import pandas as pd
from scipy import stats

class ResultsFromTest(object):
	def Pearson(self, x, y):
		return stats.pearsonr(x,y)

	def Spearman(self, x, y):
		return stats.spearmanr(x,y)

	def Kendalltau(self, x , y):
		return stats.kendalltau(x,y)

	def MannWhitney(self, x, y):
		return stats.mannwhitneyu(x,y)

	def Kolmogorov(self, x):
		return stats.kstest(x,'norm')

	def Shapiro(self, x):
		return stats.shapiro(x)

	def T_test(self, x, y):#8
		return stats.ttest_ind(x,y)
