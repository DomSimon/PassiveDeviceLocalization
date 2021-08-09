#by Dominic Simon, TU Berlin, 2020

from __future__ import division 
from tkinter import *
from sklearn.datasets import make_blobs
import math
import csv
import pandas as pd
import numpy as np
import sys
from ds_functions import *
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

#Parameters that has been found
threshold = 12 #difference of dB Values to be recognized
dbis_eps = 55
dbis_min_samples = 2

### Extended configuration for 
###appled offset from antenna dierections in 2.8 degree steps
tx_offset = 0 
rx_offset = 0 


##### Modes 
debug_mode = 0

my_window = Tk()
ds_canvas = Canvas(my_window, width=1200, height=1200)
ds_canvas.grid(row=1,column=0)
background = PhotoImage(file = "background.png")
ds_canvas.create_image(0, 0, image=background, anchor=NW)
ds_draw_top_view(ds_canvas)


# Select two Sessions, parameter acquesition. 
# cal_session: session number of empty room
# loc_session: session with subjects
if len(sys.argv)==3:
	cal_session = int(sys.argv[1])
	loc_session = int(sys.argv[2])
	print("Starting device-free passive localization with calibration session "+str(cal_session)+". Searching in session "+str(loc_session)+" started.")

else:
	print("ATTENTION wrong format. Session numbers invalid. Use for Example: python3 tg_localization 1 22 (for reference session 1 with detection in session 22.) You see now session 1 with sessoin 22")
	cal_session = 1
	loc_session = 22



# df1 represents the calibration session, df2 represents a specific measurement session.
df1 = pd.read_csv("measurements/"+str(cal_session)+" - Empty Room "+str(cal_session)+"/normalized_results.csv")

if loc_session <3:
	df2 = pd.read_csv("measurements/"+str(loc_session)+" - Empty Room "+str(loc_session)+"/normalized_results.csv")
	df2_positions = pd.read_csv("measurements/"+str(loc_session)+" - Empty Room "+str(loc_session)+"/positions.csv")
else:
	df2 = pd.read_csv("measurements/"+str(loc_session)+" Sounding/normalized_results.csv")
	df2_positions = pd.read_csv("measurements/"+str(loc_session)+" Sounding/positions.csv")
# Ground truth acquisition for plotting.

real_locations = df2_positions.values.tolist() #ground truth for  
#print(real_locations)

# Sessions are sorted to be usable with radiation angles
# many values are need to be shifted and need inverted dicections
# generated objects are now 0-63 for -45° to 45° on TX and RX
df1_sorted = generate_sorted_df(df1)
df2_sorted = generate_sorted_df(df2)

# Difference of these two Sessions is calculated
df3_diff = difference_of(df1_sorted,df2_sorted)

# Until this point every row of the dataset has now TX-Beam, RX_Beam and Path loss
# We have a approximation of the angles between the two antenna arrays and can now
# draw the parameters to absolute x,y positions.
df4_xy_param = ds_calculate_xy_positions(df3_diff, tx_offset, rx_offset, ds_canvas, threshold)

# The Points can now be plotted to the Canvas
estimated_locations = DBIS(df4_xy_param, ds_canvas, dbis_eps, dbis_min_samples)





# This Function draws the real Locations into Canvas
for i in range(0,len(df2_positions)):
	x_punkt_1 = ((df2_positions.x[i]/200)*800)+100
	y_punkt_1 = ((df2_positions.y[i]/200)*800)+500
	draw_circle(x_punkt_1, y_punkt_1,10, 'black', ds_canvas)
	ds_canvas.create_text(x_punkt_1, y_punkt_1, text=(i+1), font=("Arial", 20), fill="white")
	ds_canvas.create_text(150, (100+15*i),anchor=NW, text=("Ground truth provided in positions.csv Object "+str((i+1))+" at x:"+str(df2_positions.x[i])+"cm, y="+str(df2_positions.y[i])+"cm, (black circle)"), font=("Arial", 10))
	#ds_canvas.create_text(500, 900, text=1, font=("Arial", 20), fill="white")

#ds_canvas.create_text(420,500,anchor=NW, text="smallest Pathloss (on LOS) is "+str(int(path_loss_min))+" dB", font=("Arial", 10))
#ds_canvas.create_text(100,50,anchor="w", text="path_loss der Leermessung", font=("Arial", 16))
ds_canvas.create_text(150,40,anchor="w", text="Threshold:"+str(threshold)+" (All measurements with a difference smaller then "+str(threshold)+" are rejected)", font=("Arial", 16))
ds_canvas.create_text(150,20,anchor="w", text="Device-free passive localization using Millimeter Wave Radios by Dominic Simon)", font=("Helvetica", 20), fill="black")

my_window.mainloop()
