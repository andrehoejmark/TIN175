"""
This script requires Python 3.X or later in order to run.

A script to generate a merged dataset CSV file from separate SMHI data CSV files.
"""

import os
import os.path
import sys
import csv
import argparse
from datetime import datetime

TIMESTAMP_FORMAT_STRING = "%Y-%m-%d %H:%M:%S"
TIMESTAMP_PAST = "0000-00-00 00:00:00"
TIMESTAMP_FUTURE = "9999-12-31 23:59:59"

MIN_TIME = "0001-01-01"
MAX_TIME = "9999-12-31"

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
  except:
    print("Unable to find the file %s." % name)
  return result

def writeCSV(name, rows, columns):
  with open(name, mode = 'w') as file:
    writer = csv.writer(file, delimiter = ';', quotechar='"', quoting = csv.QUOTE_MINIMAL)
    header_row = ["Datum", "Tid (UTC)"]
    for column in columns:
      header_row.append(column)
    writer.writerow(header_row)
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
    i = 0
    l = len(set)
    while i < l:
      row = set[i]
      timestamp = getRowTimestamp(row)
      if timestamp < MIN_TIME:
        i = i + 1
      else:
        break
    indices.append(i)
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
        if (timestamp < MAX_TIME
          and timestamp < old_timestamp):
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

def parseCities(city_names, folders, columns, max_len):
  paths = map(lambda x : x + "/", folders)
  results = []
  for name in city_names:
    for column in columns:
      file_names = map(lambda path : path + column + "_" + name + ".csv", paths)
      found_file = False
      for file_name in file_names:
        if os.path.isfile(file_name):
          print("Processing city: %s from \"%s\" with the columns %s" % (name, file_name, str(columns)))
          partial = readCSV(file_name, max_len = max_len)
          print("Loaded column %s with %d rows." % (column, len(partial)))
          found_file = True
          break
      if not found_file:
        print("Unable to find the file (make sure you have addded the folder to the paths list): %s" % name)
        break
      results.append(partial)
  return mergeCSV(results)

def parseArguments(args):
  global MIN_TIME
  global MAX_TIME
  input_city = []
  output_path = "merged.csv"
  length = len(args)
  input_max_len = 500000
  parser = argparse.ArgumentParser(description = "A CSV file merging script.")
  parser.add_argument('columns', metavar = 'column' , type = str, nargs = '+')
  parser.add_argument('--city', dest = 'cities', help = 'add an additional city for which to search for weather data.', action = 'append', type = str)
  parser.add_argument('--max', dest = 'max_rows', help = 'set the maximum number of input rows read from the CSV files.', action = 'append', type = int)
  parser.add_argument('--folder', dest = 'folder', help = 'specify a folder path for input CSV files.', action = 'append', type = str)
  parser.add_argument('--output', dest = 'output', help = 'specify an output CSV file.', action = 'append', type = str)
  parser.add_argument('--start', dest = 'start', help = 'set a starting timestamp in Y-m-d format.', action = 'append', type = str)
  parser.add_argument('--stop', dest = 'stop', help = 'set a ending timestamp in Y-m-d format.', action = 'append', type = str)
  args = parser.parse_args()
  if args.start:
    MIN_TIME = args.start[-1]
  if args.stop:
    MAX_TIME = args.stop[-1]
  try:
    MIN_TIME = datetime.strptime(MIN_TIME, "%Y-%m-%d")
  except:
    print("Malformed start date.")
    return
  try:
    MAX_TIME = datetime.strptime(MAX_TIME, "%Y-%m-%d")
  except:
    print("Malformed stop date.")
    return
  if len(args.cities) == 0:
    print("Needs at least one specified city.")
  elif len(args.columns) >= 1:
    if args.output and len(args.output) > 0:
      output_path = args.output[-1]
    if not args.folder:
      args.folder = ["."]
    if os.path.isfile(output_path):
      print("The file \"%s\" already exists, please remove it or change the output name (using the option \"--output\")." % output_path)
    else:
      output = parseCities(args.cities, args.folder, args.columns, input_max_len)
      header_columns = []
      for city in args.cities:
        for column in args.columns:
          header_columns.append(city + " " + column)
      writeCSV(output_path, output, header_columns)
  else:
    print("Missing columns.")

parseArguments(sys.argv[1:])
