import mysql.connector

dataBase = mysql.connector.connect(
    host = 'localhost',
    user = 'root',
    passwd = 'root'
)

cursorObject = dataBase.cursor()

#create a database
cursorObject.execute("CREATE DATABASE RollCall_database")
print("database created")