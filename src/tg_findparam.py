

from __future__ import division 
from tkinter import *
from sklearn.datasets import make_blobs
import math
import csv
import pandas as pd
import numpy as np
from ds_functions import *
from accuracy import *
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

#Parameters to be found
threshold = 15 #difference of dB Values to be recognized
dbis_eps = 30
dbis_min_samples = 3


##### Modes 
debug_mode = 0
findparameters_mode = 1

my_window = Tk()
ds_canvas = Canvas(my_window, width=1200, height=1200)
ds_canvas.grid(row=1,column=0)
background = PhotoImage(file = "background.png")
ds_canvas.create_image(0, 0, image=background, anchor=NW)
ds_draw_top_view(ds_canvas)



df1 = pd.read_csv("measurements/1 - Empty Room 1/normalized_results.csv")
df1_sorted = generate_sorted_df(df1)

tx_offset = 0 
rx_offset = 0 

### First step: Data of all combinations will be acquired and saved to results-iteration.csv
if (findparameters_mode==1):
	results_iteration = pd.DataFrame(columns=['Session','threshold','dbscan_eps','dbscan_min_samples','lm_cerr','lm_Pfa','lm_Pd','lm_locError'])
	print("Start with parameter testing and accuracy calculation. Data processing can take some time. 25536 combinations are analyzed.")
	# This function will draw the Points of the path_loss difference
	# onto the Canvis. Calibration parameters are used to set the Points on positions
	for number in range (3, 16): #3,4,5,...14,15
		df2 = pd.read_csv("measurements/"+str(number)+" Sounding/normalized_results.csv")
		df2_positions = pd.read_csv("measurements/"+str(number)+" Sounding/positions.csv")
		real_locations = df2_positions.values.tolist()
		df2_sorted = generate_sorted_df(df2)
		df3_diff = difference_of(df1_sorted,df2_sorted)
		for threshold in range (6, 20): #difference of dB Values to be recognized
			df4_xy_param = ds_calculate_xy_positions(df3_diff, tx_offset, rx_offset, ds_canvas, threshold)
			for dbscan_eps_multiplicator in range (1,20): # is multiplied by 5. range(1,20) means eps=(5,10,15,...,85,90,95)
				dbscan_eps = dbscan_eps_multiplicator*5 #to get(5cm-95cm)
				for dbscan_min_samples in range (2,10):
					estimated_locations = DBIS(df4_xy_param, ds_canvas, dbscan_eps, dbscan_min_samples)
					#print(real_locations)
					#print(estimated_locations)
					lm = LocalizationMetrics(real_locations, estimated_locations)
					lm.calculateMetrics()
					
					#print(lm.locError)
					results_iteration=results_iteration.append({'Session': number, 'threshold': threshold,'dbscan_eps': dbscan_eps,'dbscan_min_samples': dbscan_min_samples, 'lm_cerr':lm.cerr, 'lm_Pfa':lm.Pfa, 'lm_Pd':lm.Pd,'lm_locError':lm.locError}, ignore_index=True)
		results_iteration.to_csv(r'results_iteration.csv', index = False)
		print("Session "+str(number)+" completed.")
print("End with Iteration")

### Second step: Average values for each combination are generatedfrom results-iteration.csv and saved to results_iteration_sorted.csv

df_results = pd.read_csv("results_iteration.csv")
results_iteration = pd.DataFrame(columns=['threshold','dbscan_eps','dbscan_min_samples','lm_cerr_avg','lm_Pfa_avg','lm_Pd_avg','lm_locError_avg'])
length=len(df_results.index)
    

for threshold in range (6, 20): #difference of dB Values to be recognized
	for dbscan_eps_multiplicator in range (1,20): 
		dbscan_eps = dbscan_eps_multiplicator*5 #to get(5cm-100cm)
		print("collecting data from threshold:"+str(int(threshold))+" dbscan_eps:"+str(int(dbscan_eps)))
		for dbscan_min_samples in range (2,10):
			lm_cerr_sum=0
			lm_Pfa_sum=0
			lm_Pd_sum=0
			lm_locError_sum=0
			counter=0

			for i in range(0,length):
				if ((df_results.threshold[i]==threshold)and(df_results.dbscan_eps[i]==dbscan_eps)and(df_results.dbscan_min_samples[i]==dbscan_min_samples)):
					lm_cerr_sum=lm_cerr_sum+df_results.lm_cerr[i]
					lm_Pfa_sum=lm_Pfa_sum+df_results.lm_Pfa[i]
					lm_Pd_sum=lm_Pd_sum+df_results.lm_Pd[i]
					lm_locError_sum=lm_locError_sum+df_results.lm_locError[i]
					counter=counter+1	

			lm_cerr_avg=lm_cerr_sum/counter
			lm_Pfa_avg=lm_Pfa_sum/counter
			lm_Pd_avg=lm_Pd_sum/counter
			lm_locError_avg=lm_locError_sum/counter


			results_iteration=results_iteration.append({'threshold': threshold,'dbscan_eps': dbscan_eps,'dbscan_min_samples': dbscan_min_samples, 'lm_cerr_avg':lm_cerr_avg, 'lm_Pfa_avg':lm_Pfa_avg, 'lm_Pd_avg':lm_Pd_avg,'lm_locError_avg':lm_locError_avg}, ignore_index=True)
			results_iteration.to_csv(r'results_iteration_sorted.csv', index = False)
			




