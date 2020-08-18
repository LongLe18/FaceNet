import sqlite3 as lite
import sys
import os

# path = os.path.dirname(__file__) + "\\FaceBaseNew.db"
con = lite.connect("D:\PracticePY\Project\Study\client-server\custom-face-recognition\FaceBaseNew.db")

with con:
    #con.row_factory = lite.Row
    cur = con.cursor()
    #cur.execute("DROP TABLE IF EXISTS People")
    #cur.execute("CREATE TABLE People(Id INT, Name TEXT, Age TEXT)")
    cur.execute("SELECT * FROM People")

    # cur.execute("INSERT INTO People VALUES(2,'John','34')")
    # cur.execute("INSERT INTO People VALUES(3,'Unknown','???')")
    # rows = cur.fetchall()
    # for row in rows:
    #     print ("%s %s" % (row["Id"], row["Name"]))
    #     if row["Name"] == " Long ":
    #         print (row["Id"])
    # for row in rows:
    #     print ("%s %s" % (row["Id"], row["Name"]))

    while True:
       
        row = cur.fetchone()
         
        if row == None:
            break
        print(row)
    