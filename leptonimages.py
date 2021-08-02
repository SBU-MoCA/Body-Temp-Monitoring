"""
Stream images from the Lepton and print min/max temperature values.

From the "Examples" folder in the Lepton Windows SDK: lepton.flir.com/software-sdk/#windows-sdk
"""

import clr # needs the "pythonnet" package
import sys
import os
import time

# check whether python is running as 64bit or 32bit
# to import the right .NET dll
import platform
bits, name = platform.architecture()

if bits == "64bit":
	folder = ["x64"]
else:
	folder = ["x86"]

sys.path.append(os.path.join("..", *folder))


clr.AddReference("LeptonUVC")
from Lepton import CCI

found_device = None
for device in CCI.GetDevices():
    if device.Name.startswith("PureThermal"):
        found_device = device
        break

if not found_device:
    print("Couldn't find lepton device")
else:
    lep = found_device.Open()
    

# Streaming frames from the Lepton
clr.AddReference("ManagedIR16Filters")
from IR16Filters import IR16Capture, NewIR16FrameEvent, NewBytesFrameEvent
import numpy
from matplotlib import pyplot as plt
# Uncomment this line if running in Jupyter
#%matplotlib inline

capture = None


from collections import deque

# change maxlen to control the number of frames of history we want to keep
incoming_frames = deque(maxlen=10)
def got_a_frame(short_array, width, height):
    incoming_frames.append((height, width, short_array))

if capture != None:
    # don't recreate capture if we already made one
    capture.RunGraph()
else:
    capture = IR16Capture()
    capture.SetupGraphWithBytesCallback(NewBytesFrameEvent(got_a_frame))
    capture.RunGraph()
    
def short_array_to_numpy(height, width, frame):
    return numpy.fromiter(frame, dtype="uint16").reshape(height, width)


# Display the image
from matplotlib import cm
height, width, net_array = incoming_frames[-1]
arr = short_array_to_numpy(height, width, net_array)
plt.imshow(arr, cmap=cm.plasma)

def centikelvin_to_celsius(t):
    return (t - 27315) / 100

def to_fahrenheit(ck):
    c = centikelvin_to_celsius(ck)
    return c * 9 / 5 + 32

# get the max image temp
print("maximum temp {:.2f} ºF / {:.2f} ºC".format(
    to_fahrenheit(arr.max()), centikelvin_to_celsius(arr.max())))
# get the average image temp
print("average temp {:.2f} ºF / {:.2f} ºC".format(
    to_fahrenheit(arr.mean()), centikelvin_to_celsius(arr.mean())))