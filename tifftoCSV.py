"""
Adds CSV files for all .tiff files in a directory.
"""

from PIL import Image
import numpy as np
import csv
import os

def csvWriter(fil_name, nparray):
  example = nparray.tolist()
  with open(fil_name+'.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(example)

for name in os.listdir():
    if name[-4:] == 'tiff':
      img = np.array(Image.open(name))
      csvWriter(name[:-5], img)