########################################################################
# setup.py,
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
from distutils.core import setup
import os

class SetupConfiguration:

    def __init__(self):

        self.setupInstall()

    def setupInstall(self):

        setup(name='NONLINEAR_MODELS',
            version='1.0',
            description='Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.',
            author='David Medina',
            author_email='david.medina@cebib.cl',
            license='Open GPL 3',
            packages=['modulesNLM', 'modulesNLM.clustering_analysis', 'modulesNLM.checks_module', 'modulesNLM.supervised_learning_analysis', 'modulesNLM.supervised_learning_predicction', 'modulesNLM.utils', 'modulesNLM.graphic', 'modulesNLM.statistics_analysis', 'modulesNLM.statisticsCorroboration'],)

def main():

    setup = SetupConfiguration()
    return 0

if __name__ == '__main__':
    main()
