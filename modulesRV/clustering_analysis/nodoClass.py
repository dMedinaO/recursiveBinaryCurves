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
import time

class Nodo(object):
    def __init__(self, data):
        # Data contiene el dataFrame de cada nodo
        self.data = data
        # Marca unica para realizar enlace y edges con graphviz
        self.id = int(round(time.time() * 1000))
        # rama izq
        self.left = None
        # rama der
        self.right = None
