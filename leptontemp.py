"""
Interacting with the Lepton from Python.

From the "Examples" folder in the Lepton Windows SDK: lepton.flir.com/software-sdk
"""

#Setup Python's path for the Lepton .NET dlls
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
    
sys.path.append(os.path.join("..", *folder)
                

#Import the CCI SDK
clr.AddReference("LeptonUVC")

from Lepton import CCI
                
found_device = None #Look for a PureThermal USB device
for device in CCI.GetDevices():
    if device.Name.startswith("PureThermal"):
        found_device = device
        break

if not found_device:
    print("Couldn't find lepton device")
else:
    lep = found_device.Open()
                

#Streaming frames from the Lepton
clr.AddReference("ManagedIR16Filters")
from IR16Filters import IR16Capture, NewIR16FrameEvent, NewBytesFrameEvent #a library for grabbing 16-bit greyscale images

import numpy
from matplotlib import pyplot as plt
%matplotlib inline #displays plot results inline in Jupyter
capture = None

from collections import deque
incoming_frames = deque(maxlen=10) #change maxlen to control the size of the queue
def got_a_frame(short_array, width, height):
    incoming_frames.append((height, width, short_array))

if capture != None:
    # don't recreate capture if we already made one
    capture.RunGraph()
else:
    capture = IR16Capture()
    capture.SetupGraphWithBytesCallback(NewBytesFrameEvent(got_a_frame))
    capture.RunGraph()

"""
Re-run the following code in Jupyter to display the most recent image in the queue using matplotlib.
"""
def short_array_to_numpy(height, width, frame):
    return numpy.fromiter(frame, dtype="uint16").reshape(height, width)
                
from matplotlib import cm

height, width, net_array = incoming_frames[-1]
arr = short_array_to_numpy(height, width, net_array)
plt.imshow(arr, cmap=cm.plasma) #Shows an image in real-time from the Lepton

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
print("temp at ({},{}) is {}".format(100,50,arr(100,50)))                

                
"""
Uncomment the following to test if the camera supports TLinear and can return raw temperature values. 
"""
# try:
#     lep.rad.SetTLinearEnableStateChecked(True)
#     print("this lepton supports tlinear")
# except:
#     print("this lepton does not support tlinear")