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

def readCSV(name, wid = 3, max_len = 500000, is_multi = False):
  result = []
  try:
    with open(name) as file:
      reader = csv.reader(file, delimiter = ';', quotechar='"')
      valid_lines = False
      count = 0
      for row in reader:
        if valid_lines:
          part = row[0:wid]
          if is_multi:
            if len(row) < 5:
              print("Missing value (will default to \"-\") on row (%s) in file: %s" % (row, name))
              part.append("-")
            else:
              part.append(row[wid + 1])
          result.append(part)
          if(count >= max_len):
            break
          else:
            count = count + 1
        elif(len(row) > 0 and row[0] == "Datum"):
          valid_lines = True
  except Exception as e:
    print("Threw exception:" % e)
    sys.exit(1)
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
  print("Seeking to start timestamp...")
  columns_width = 2
  for set in results:
    i = 0
    l = len(set)
    columns_width = columns_width + len(set[0]) - 2
    # TODO: Optimize using binary search.
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
  iteration = 0
  print("Merging data rows...")
  while(True):
    iteration = iteration + 1
    if iteration > 50000:
      iteration = 0
      parsed = 0
      total = 0
      for i in range(0, len(results)):
        index = indices[i]
        parsed = parsed + index
        total = total + len(results[i])
      print("Progress: %f%% done." % ((float(parsed) / float(total)) * 100.0) )
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
      new_row = [row_date, row_time]
      for i in range(0, len(results)):
        index = indices[i]
        set = results[i]
        hit = False
        if index < len(set):
          row = set[index]
          if row[0] == row_date and row[1] == row_time:
            indices[i] = index + 1
            hit = True
            for e in row[2:]:
              new_row.append(e)
        if not hit:
          for e in row[2:]:
            new_row.append("-")
      if len(new_row) != columns_width:
        print("Column witdth exception! Got %d expected %d" % (len(new_row), columns_width))
        print(new_row)
        return output
      output.append(new_row)
    else:
      break # No more data available.
  print("END!")
  return output

def parseCities(city_names, folders, columns, max_len, multi):
  results = []
  for name in city_names:
    for column in columns:
      found_file = False
      for folder in folders:
        file_name = folder + "/" + column + "_" + name + ".csv"
        print("Looking for: %s" % file_name)
        if os.path.isfile(file_name):
          print("Processing city: %s from \"%s\" with the columns %s" % (name, file_name, str(columns)))
          partial = readCSV(file_name, max_len = max_len, is_multi = column in multi)
          results.append(partial)
          print("Loaded column %s with %d rows." % (column, len(partial)))
          found_file = True
          break
        else:
          print("Unable to find file: %s" % file_name)
      if not found_file:
        print("Unable to find the file (make sure you have addded the folder to the paths list): %s" % column)
  return mergeCSV(results)

def parseArguments(args):
  global MIN_TIME
  global MAX_TIME
  input_city = []
  output_path = "merged.csv.old"
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
  parser.add_argument('--multi', dest = 'multi', help = 'name of a column which when read will read two two data columns instead of one.', action = 'append', type = str)
  args = parser.parse_args()
  if args.start:
    MIN_TIME = args.start[-1]
  if args.stop:
    MAX_TIME = args.stop[-1]
  try:
    MIN_TIME = datetime.strptime(MIN_TIME, "%Y-%m-%d")
  except:
    print("Malformed start date.")
    sys.exit(1)
  try:
    MAX_TIME = datetime.strptime(MAX_TIME, "%Y-%m-%d")
  except:
    print("Malformed stop date.")
    sys.exit(1)
  if len(args.cities) == 0:
    print("Needs at least one specified city.")
  elif len(args.columns) >= 1:
    if args.output and len(args.output) > 0:
      output_path = args.output[-1]
    if not args.folder:
      args.folder = ["."]
    if not args.multi:
      args.multi = []
    if os.path.isfile(output_path):
      print("The file \"%s\" already exists, please remove it or change the output name (using the option \"--output\")." % output_path)
    else:
      output = parseCities(args.cities, args.folder, args.columns, input_max_len, args.multi)
      header_columns = []
      for city in args.cities:
        for column in args.columns:
          header_columns.append(city + " " + column)
          if column in args.multi:
            header_columns.append(city + " " + column + " #2")
      writeCSV(output_path, output, header_columns)
  else:
    print("Missing columns.")

parseArguments(sys.argv[1:])
