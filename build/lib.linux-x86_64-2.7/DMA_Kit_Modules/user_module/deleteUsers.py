'''
script que tiene la responsabilidad de eliminar al usuario cuando su cuenta esta cancelada, previo a ello, obtiene todos los jobs
asociados a su ID, los elimina de la base de datos y elimina todos los recursos asociados a este...
'''

from modulesProject.dataBase_module import ConnectDataBase
from modulesProject.dataBase_module import HandlerQuery
import os

class deleteUsers(object):

    def __init__(self):

        #hacemos las instancias de conexion y handler de la bd
        self.connect = ConnectDataBase.ConnectDataBase()
        self.handler = HandlerQuery.HandlerQuery()

    #metodo que permite obtener todos los usuarios en estado CANCELED
    def getCanceledUsers(self):

        query = "select user.iduser from user where user.statusUser = 'CANCELED'"

        self.connect.initConnectionDB()
        listResponse = self.handler.queryBasicDataBase(query, self.connect)
        self.ListUserRemove = []

        for element in listResponse:
            self.ListUserRemove.append(element[0])

        self.connect.closeConnectionDB()

    #metodo que permite obtener los jobs del usuario
    def getJobsUser(self, userID):

        query = "select job.idjob from job where job.user = %s" % userID

        self.connect.initConnectionDB()
        listResponse = self.handler.queryBasicDataBase(query, self.connect)
        listJobs = []

        for element in listResponse:
            listJobs.append(element[0])

        self.connect.closeConnectionDB()

        return listJobs

    #metodo que permite eliminar el job y remover el directorio
    def deleteJob(self, job, user):

        #eliminamos el job de la bd
        query = "delete from job where job.idjob = %s" % job
        self.connect.initConnectionDB()
        self.handler.insertToTable(query, self.connect)

        #tambien eliminamos los data set asociados...
        query = "delete from dataSet where dataSet.job = %s" % job
        self.connect.initConnectionDB()
        self.handler.insertToTable(query, self.connect)
        self.connect.closeConnectionDB()


    #metodo que permite remover la carpeta del usuario...
    def removeAreaJob(self, user):

        #eliminamos el directorio
        command = "rm -rf /var/www/html/smartTraining/dataStorage/%s/" % user
        os.system(command)
