import sqlite3

connection=sqlite3.connect('Student.db')
db_cursor=connection.cursor()#Cursor object to perform operations on the database

table="""
    CREATE TABLE STUDENT
    (
    roll_no INTEGER PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    class VARCHAR(10) NOT NULL,
    GRADE VARCHAR(10) NOT NULL
    )
"""

db_cursor.execute(table)

db_cursor.execute('''INSERT INTO STUDENT VALUES (1,'Varun','S4','A')''')
db_cursor.execute('''INSERT INTO STUDENT VALUES (2,'Arun','S5','A+')''')
db_cursor.execute('''INSERT INTO STUDENT VALUES (3,'Soman','S4','B')''')
db_cursor.execute('''INSERT INTO STUDENT VALUES (4,'Mohan','S6','A+')''')
db_cursor.execute('''INSERT INTO STUDENT VALUES (5,'Gopal','S4','B+')''')

print('Inserted Records are')
data=db_cursor.execute('''SELECT * FROM STUDENT''')
for row in data:
    print(row)

connection.commit()
connection.close()