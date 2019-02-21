
import csv

class smhiCSV:
  def __init__(self):
    self.header = []
    self.data = []

def loadCSV(name):
  result = smhiCSV()
  try:
    with open(name) as file:
      reader = csv.reader(file, delimiter = ';', quotechar='"')
      result.header = next(reader)
      for row in reader:
        result.data.append(row)
  except Exception as e:
    print("Failed to open file: %s" % name)
  return result

def loadInterpolatedCSV(name):
  return interpolateCSV(loadCSV(name))

def interpolateCSV(smhi):
  data = smhi.data
  num_rows = len(data)
  for i in range(2, len(smhi.header)):
    last_valid_data = None
    last_invalid_row = -1
    last_valid_row = -1
    is_invalid = False
    for j in range(0, num_rows):
      row = data[j]
      cell = row[i]
      if cell is "-":
        if not is_invalid:
          is_invalid = True
          last_invalid_row = j
      else:
        if is_invalid:
          is_invalid = False
          if last_valid_row >= 0:
            last_valid_data = float(data[last_valid_row][i])
          else:
            print("Unable to interpolate data, the first row value may not be invalid. Will use default start value 0.0")
            last_valid_row = 0
            last_valid_data = 0
          from_value = last_valid_data
          to_value = float(data[j][i])
          step = (to_value - from_value) / float(j - last_valid_row)
          print("Interpolating column %d from row %d to row %d" % (i, last_valid_row, j))
          for k in range(last_invalid_row, j):
            data[k][i] = from_value + (k - last_invalid_row) * step
            print(data[k][i])
        last_valid_row = j
  return data
