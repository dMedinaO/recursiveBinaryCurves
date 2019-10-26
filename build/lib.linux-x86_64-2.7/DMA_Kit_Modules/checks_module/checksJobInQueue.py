########################################################################
# checksJobsInQueue.py,
#
# Checks jobs in queue.
# If the jobs exists then they are executed. Job status is changed to "Processing".
# An email is sent with the change in the job status.
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

from modulesProject.dataBase_module import ConnectDataBase
from modulesProject.dataBase_module import HandlerQuery
from modulesProject.utils import sendEmail
import os

class checkJobs(object):

    def __init__(self):

        #hacemos las instancias de conexion y handler de la bd
        self.connect = ConnectDataBase.ConnectDataBase()
        self.handler = HandlerQuery.HandlerQuery()

    #metodo que permite obtener los jobs en estado init...
    def getJobsInit(self):

        self.connect.initConnectionDB()
        query = "select job.idjob from job where job.statusJob = 'START' AND job.tipo_job like '%queue%'"
        listResponse = self.handler.queryBasicDataBase(query, self.connect)

        self.listJobs = []
        for element in listResponse:
            self.listJobs.append(element[0])

        self.connect.closeConnectionDB()

    #metodo que permite obtener la informacion del usuario...
    def getUserInfoByJob(self, jobID):

        query = "select user.iduser, user.nameUser, user.emailUser, job.tipo_job from user join job on (user.iduser = job.user) where job.idjob = %s" % jobID
        self.connect.initConnectionDB()
        listResponse = self.handler.queryBasicDataBase(query, self.connect)

        self.connect.closeConnectionDB()
        return int(listResponse[0][0]), str(listResponse[0][1]), str(listResponse[0][2]), str(listResponse[0][3])#retornamos el nombre de usuario, el email y el tipo de job

    #metodo que permite obtener el nombre del set de datos...
    def getNameDataSet(self, job):

        query = "select job.nameDataset from job where job.idjob = %s" %job
        self.connect.initConnectionDB()
        listResponse = self.handler.queryBasicDataBase(query, self.connect)

        self.connect.closeConnectionDB()
        return listResponse[0][0]#retornamos el nombre del set de datos

    #metodo que permite recibir un job y hacer la ejecucion del job, cambia estado y notifica el cambio via correo electronico
    def execJob(self, job):

        nameDataset = self.getNameDataSet(job)
        iduser, nameUser, emailUser, tipo_job = self.getUserInfoByJob(job)

        #hacemos la actualizacion del job en el servidor
        query = "update job set job.statusJob = 'PROCESSING', job.modifiedJob=NOW() where job.idjob = %s" % job
        self.connect.initConnectionDB()
        self.handler.insertToTable(query, self.connect)

        #hacemos la notificacion al usuario...
        body = "Dear %s.\nThe job with ID: %s has been update to status: PROCESSING. It will notify by email when job finish.\nBest Regards, SmartTraining Team" % (nameUser, job)
        emailData = sendEmail.sendEmail('smarttrainingserviceteam@gmail.com', emailUser, "Change status in job "+ str(job), body, 'smart123ewq')
        emailData.sendEmailUser()
        self.connect.closeConnectionDB()

        dataSet = "/var/www/html/smartTraining/dataStorage/%s/%s/%s" % (iduser, job, nameDataset)
        if tipo_job == "queue-PREDICTION":#prediction launcher

            command = "python /var/www/html/smartTraining/model/launcherFullProcessPrediction.py %s %s %s /var/www/html/smartTraining/dataStorage/ %s %s &" % (iduser, job, dataSet, emailUser, nameUser)
            os.system(command)
            print command
        else:#classification launcher
            command = "python /var/www/html/smartTraining/model/launcherFullProcessClassification.py %s %s %s /var/www/html/smartTraining/dataStorage/ %s %s &" % (iduser, job, dataSet, emailUser, nameUser)
            os.system(command)
            print command
