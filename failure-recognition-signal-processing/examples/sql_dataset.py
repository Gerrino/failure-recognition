from pathlib import Path
 
import sqlite3
from sqlite3 import OperationalError

conn = sqlite3.connect('csc455_HW3.db')
c = conn.cursor()
file_path = Path("C:\git/failure_recognition/failure-recognition-signal-processing\examples/buehler_ts_timeseries_me.sql")
print("exists", file_path.exists())

def executeScriptsFromFile(filename):
    # Open and read the file as a single buffer
    with open(filename, 'r') as fd:
        sqlFile = fd.read()

    

    # all SQL commands (split on ';')
    sqlCommands = sqlFile.split(';')

    print(sqlCommands[1:100])

    # Execute every command from the input file
    for command in sqlCommands:
        # This will skip and report errors
        # For example, if the tables do not yet exist, this will skip over
        # the DROP TABLE commands
        try:
            c.execute(command)
            print(command)
        except OperationalError as msg:
            print("Command skipped: ", str(msg))

executeScriptsFromFile(file_path)
pass