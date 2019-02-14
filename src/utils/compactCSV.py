"""
This script requires Python 3.X or later in order to run.

A script to generate a merged dataset CSV file from separate SMHI data CSV files.
"""

import os
import sys
import csv
from datetime import datetime

TIMESTAMP_FORMAT_STRING = "%Y-%m-%d %H:%M:%S"
TIMESTAMP_FUTURE = "9999-12-31 23:59:59"

def readCSV(name, wid = 3, max_len = 500000):
  result = []
  try:
    with open(name) as file:
      reader = csv.reader(file, delimiter = ';', quotechar='"')
      valid_lines = False
      for row in reader:
        if valid_lines:
          result.append(row[0:wid])
          if(len(result) >= max_len):
            break
        elif(len(row) > 0 and row[0] == "Datum"):
          valid_lines = True
  except FileNotFoundError:
    print("Unable to find the file %s." % name)
  return result

def writeCSV(name, rows):
  with open(name, mode = 'w') as file:
    writer = csv.writer(file, delimiter = ';', quotechar='"', quoting = csv.QUOTE_MINIMAL)
    writer.writerow(["Datum", "Tid (UTC)"])
    for row in rows:
      writer.writerow(row)

def getRowTimestamp(row):
  return datetime.strptime(row[0] + " " + row[1], TIMESTAMP_FORMAT_STRING)

def mergeCSV(results):
  # This expectes the values to be sorted in occurance (by data and time) and that the
  # first two columns to hold those two values.
  output = []
  indices = []
  for set in results:
    indices.append(0)
  future_timestamp = datetime.strptime(TIMESTAMP_FUTURE, TIMESTAMP_FORMAT_STRING)
  row_date = ""
  row_time = ""
  while(True):
    old_timestamp = future_timestamp
    has_more = False
    # Find earliest time on stack.
    for i in range(0, len(results)):
      index = indices[i]
      set = results[i]
      if index < len(set):
        row = set[index]
        timestamp = getRowTimestamp(row)
        if timestamp < old_timestamp:
          row_date = row[0]
          row_time = row[1]
          old_timestamp = timestamp
          has_more = True
    # Collect relevant values and add to a merged row.
    if has_more:
      # print(old_timestamp)
      new_row = [row_date, row_time]
      for i in range(0, len(results)):
        index = indices[i]
        set = results[i]
        if index < len(set):
          row = set[index]
          if row[0] == row_date and row[1] == row_time:
            indices[i] = index + 1
            new_row.append(row[2])
          else:
            new_row.append("-")
      output.append(new_row)
    else:
      break # No more data available.
  return output

def parseCity(name, folder, columns, max_len):
  path = folder + "/"
  print("Processing city: %s from \"%s\" with the columns %s" % (name, path, str(columns)))
  results = []
  for column in columns:
    partial = readCSV(path + column + "_" + name + ".csv", max_len = max_len)
    print("Loaded column %s with %d rows." % (column, len(partial)))
    results.append(partial)
  return mergeCSV(results)

def parseArguments(args):
  input_city = None
  output_path = "merged.csv"
  input_folder = "."
  length = len(args)
  input_columns = []
  input_max_len = 500000
  if length >= 1:
    input_city = args[0]
    for arg in args[1:]:
      if arg[0:7] == "folder=":
        input_folder = arg[7:]
      elif arg[0:7] == "output=":
        output_path = arg[7:]
      elif arg[0:4] == "max=":
        try:
          input_max_len = int(arg[4:])
        except:
          print("Parameter to 'max=' is not an integer.")
          return
      else:
        input_columns.append(arg)
    if os.path.isfile(output_path):
      print("The file \"%s\" already exists, please remove it or change the output name (using the option \"output=...\")." % output_path)
    else:
      output = parseCity(input_city, input_folder, input_columns, input_max_len)
      writeCSV(output_path, output)
  else:
    print("""
Expected the first argument to be a name for weather data in a city.
---
USAGE: city-name [OPTIONS] [COLUMNS]
---
OPTIONS:
folder=... : CSV resource folder path
output=... : output CSV file
max=... : max read input table length
---
You will need to provide the CSV data files named as \"{column-name}_{city-name}.csv\" and
place them in the folder specified by the \"folder=...\" option (the default is \"folder=.\"
, the current directory). The output CSV file can be specified using the \"output=...\"
option, the default is \"output=merged.csv\". When using the \"max=...\" only the number of
specified rows will be read from each input file (default is \"max=500'000\").
""")

parseArguments(sys.argv[1:])
