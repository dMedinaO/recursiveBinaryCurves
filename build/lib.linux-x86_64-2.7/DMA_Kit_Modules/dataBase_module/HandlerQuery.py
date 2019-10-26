import ConnectDataBase

#class for insert elements in data base
class HandlerQuery:

	#method for insert, delete or update element in table
	def insertToTable(self, query, ConnectionDB):
		ConnectionDB.cursor.execute(query)
		ConnectionDB.Conex.commit()
		#self.ConnectionDB.cursor.close()

	#method for generate basic query to data base
	def queryBasicDataBase(self, query, ConnectionDB):
		ConnectionDB.cursor.execute(query)

		collection_id = []
		for element in ConnectionDB.cursor:

			collection_id.append(element)

		return collection_id
