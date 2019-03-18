
#
# Author: Felix Hulthén.
#

"""
Loading script for opening CSV files for automated network construction and testing.
"""

import math
import csv
import numpy

class Config:
  """
  Container class for data loaded from CSV files containing parameters for
  automated recurrent neural network training.
  """
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
  """
  Container class for holding a set of 'Config' objects for automated testing,
  e.g. a class for running multiple automated test sessions.
  """
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
  """
  A container class for loaded merged SMHI CSV files. Contains row headers,
  row data and modification functions.
  """
  def __init__(self, use_full_timestamp=True):
    self.header = []
    self.data = []
    self.use_full_timestamp = use_full_timestamp
  def shiftDataColumns(self, steps):
    """
    shiftDataColumns : Will shift the data rows a set number of steps, which
                       will alter the number of available rows. This is used to
                       simulate forecasting 'x steps' into the future in the simulation.
      steps : The number of steps to shift. A row that has been shifted out of the
              original length with be removed.
    """
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
    """
    drop : Will drop the first 'step' rows.
    """
    if steps > 0:
      self.data = self.data[steps:]
  def pop(self, steps):
    """
    pop : Will pop the last 'step' rows.
    """
    if steps > 0:
      self.data = self.data[:-steps]
  def valuesWithDate(self):
    """
    valuesWithDate : Expands and separates the date and time fields
                     into their respective columns, depending on the
                     state of 'use_full_timestamp'. If 'use_full_timestamp'
                     is set to False, then the year, minutes and seconds
                     fields are removed. Otherwise all fields are used.
    """
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
    """
    valuesWithoutDate : 
    """
    copy = []
    for row in self.data:
      copy.append(row[2:])
    return numpy.array(copy)
  def values(self):
    return numpy.array(self.data)

def toCellNum(n):
  """
  toCellNum : attempts to convert 'n' into a floating point number. Otherwise
              returns the original value of 'n'.
  """
  if n is "-":
    return n
  else:
    return float(n)

def loadCSV(name, convert = True):
  """
  loadCSV : Loads a raw SMHI CSV file.
    convert : If set to 'True' all row values apart from the date and time columns
              are converted into floating point numbers.
  """
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
  """
  loadConfigCSV : Load a SMHI CSV config file.
    name : file path to the config file.
  """
  result = loadCSV(name, convert = False)
  if result:
    return ConfigCSV(result)
  else:
    return None

def loadInterpolatedCSV(name):
  """
  loadInterpolatedCSV : Load a SMHI CSV dataset file and interpolate missing values.
    name : file path to the dataset.
  """
  result = loadCSV(name)
  if result:
    return interpolateCSV(result)
  else:
    return None

def interpolateCSV(smhi):
  """
  interpolateCSV : Applies linear interpolation to missing values.
    smhi : The original SMHI object.
  """
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
  """
  separateCSV : Separates the columns in a SMHI object into two separate SMHI objects.
    smhi : the original object.
    frm : the start splice index
    to : the end splice index
  """
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
  smhi.header.pop(idx)
  for row in smhi.data:
    row.pop(idx)
