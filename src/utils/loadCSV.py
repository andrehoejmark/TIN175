
import math
import csv
import numpy

class Config:
  def __init__(self, row):
    self.time_shift = int(row[0])
    self.training_split = float(row[1])
    self.start_learning_rate = float(row[2])
    self.loss_factor = float(row[3])
    self.min_learning_rate = float(row[4])
    self.learning_patience = int(row[5])
    self.num_epochs = int(row[6])
    self.steps_per_epoch = int(row[7])
    self.early_stop_patience = int(row[8])
    self.warmup_steps = int(row[9])
    self.num_gru = int(row[10])
    self.activation_function = row[11]
    self.id = row[12]
    self.loss = row[13]
    self.network_type = row[14]
    self.read_batch_size = int(row[15])
    self.outputs = int(row[16])
class ConfigCSV:
  def __init__(self, csv):
    self.rows = csv.data
  def getConfig(self, nid):
    for row in self.rows:
      if row[12] == str(nid):
        return Config(row)
      elif row[0] == "STOP":
        return None
    return None

class smhiCSV:
  def __init__(self, use_full_timestamp=True):
    self.header = []
    self.data = []
    self.use_full_timestamp = use_full_timestamp
  def shiftDataColumns(self, steps):
    copy = self.data[:]
    l = len(copy)
    broken_row = []
    for i in self.header:
      broken_row.append(math.nan)
    for i in range(0, l):
      if i - steps < 0 or i - steps >= l:
        self.data[i] = broken_row
      else:
        self.data[i] = copy[(i - steps) % l]
  def drop(self, steps):
    if steps > 0:
      self.data = self.data[steps:]
  def pop(self, steps):
    if steps > 0:
      self.data = self.data[:-steps]
  def valuesWithDate(self):
    copy = []
    for row in self.data:
      date = row[0].split("-")
      time = row[1].split(":")
      if self.use_full_timestamp:
        copy.append(
          [int(date[0]), int(date[1]), int(date[2])] +
          [int(time[0]), int(time[1]), int(time[2])] + row[2:])
      else:
        copy.append(
          [int(date[1]), int(date[2])] +
          [int(time[0])] + row[2:])
    return numpy.array(copy)
  def valuesWithoutDate(self):
    copy = []
    for row in self.data:
      copy.append(row[2:])
    return numpy.array(copy)
  def values(self):
    return numpy.array(self.data)

def toCellNum(n):
  if n is "-":
    return n
  else:
    return float(n)

def loadCSV(name, convert = True):
  try:
    result = smhiCSV()
    with open(name) as file:
      reader = csv.reader(file, delimiter = ';', quotechar='"')
      result.header = next(reader)
      for row in reader:
        nrow = row[0:2]
        for n in row[2:]:
          if convert:
            nrow.append(toCellNum(n))
          else:
            nrow.append(n)
        result.data.append(nrow)
    print("Loaded %d rows from %s" % (len(result.data), name))
    return result
  except Exception as e:
    print("Failed to open file: %s (error: %s)" % (name, e))
  return None

def loadConfigCSV(name):
  result = loadCSV(name, convert = False)
  if result:
    return ConfigCSV(result)
  else:
    return None

def loadInterpolatedCSV(name):
  result = loadCSV(name)
  if result:
    return interpolateCSV(result)
  else:
    return None

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
          # print("Interpolating column %d from row %d to row %d" % (i, last_valid_row, j))
          for k in range(last_invalid_row, j):
            data[k][i] = from_value + (k - last_invalid_row) * step
        last_valid_row = j
  return smhi

def separateCSV(smhi, frm, wid):
  to = frm + wid
  smhi_a = smhiCSV()
  smhi_b = smhiCSV()
  length = len(smhi.data)
  smhi_a.header = smhi.header[0:frm]
  for row in smhi.data:
    smhi_a.data.append(row[0:frm])
  end = len(smhi.data[0])
  smhi_a.header = smhi_a.header + smhi.header[to:end]
  for j in range(0, len(smhi.data)):
    smhi_a.data[j] = smhi_a.data[j] + smhi.data[j][to:end]
  # smhi_b.header = smhi.header[0:2] + smhi.header[frm:to]
  smhi_b.header = smhi.header[frm:to]
  for row in smhi.data:
    #smhi_b.data.append(row[0:2] + row[frm:to])
    smhi_b.data.append(row[frm:to])
  return (smhi_a, smhi_b)

def deleteCSVColumn(smhi, idx):
  print("DEL %d LEN %d" % (idx, len(smhi.header)))
  smhi.header.pop(idx)
  for row in smhi.data:
    print(len(row))
    row.pop(idx)
