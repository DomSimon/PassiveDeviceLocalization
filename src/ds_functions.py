#by Dominic Simon, TU Berlin, 2020

from __future__ import division 
from tkinter import *
import math
import csv
import pandas as pd


def ds_calculate_xy_positions(df3_xy, tx_offset, rx_offset, canvas_d, threshold_d):
    df_xy = pd.DataFrame(columns=['x','y','path_loss'])
    length_xy = len(df3_xy.index)
    for i in range(0,length_xy):
        rx_beam_d = df3_xy.tx_beam[i]-rx_offset
        tx_beam_d = df3_xy.rx_beam[i]-tx_offset
        pl_d = df3_xy.path_loss[i]
        tx_degree = (((tx_beam_d)/63)*90)-45 # From beam 0-63 to -45 up to + 45 degree
        tx_elevation = math.tan(math.radians(tx_degree)) #elevation as XY-factor
        position_x_TX = 100
        position_y_TX = 500
        tx_x1 = position_x_TX
        tx_y1 = position_y_TX
        tx_x2 = position_x_TX+(800)
        tx_y2 = position_y_TX+(800*tx_elevation)
        TX_Strahl = line([tx_x1,tx_y1], [tx_x2,tx_y2]) #needed for intersection

        rx_degree = ((((rx_beam_d)/63)*90)-45) 
        rx_elevation = math.tan(math.radians(rx_degree))
        position_x_RX = 900
        position_y_RX = 500
        rx_x1 = position_x_RX
        rx_y1 = position_y_RX
        rx_x2 = position_x_RX-(800)
        rx_y2 = position_y_RX+(800*rx_elevation)
        RX_Strahl = line([rx_x1,rx_y1], [rx_x2,rx_y2]) #needed for intersection


        p_size = pl_d/4
        p_size_to_print = pl_d


        P = intersection(RX_Strahl, TX_Strahl)
        if bool(P) and P[0]>100 and P[0]<900 and pl_d>threshold_d:
          x1, y1 = (P[0] - p_size), (P[1] - p_size)
          x2, y2 = (P[0] + p_size), (P[1] + p_size)
          canvas_d.create_oval(x1, y1, x2, y2, fill='grey90')
          canvas_d.create_text(P[0], P[1], text=p_size_to_print, font=("Arial", 8))
          xa1 = P[0]
          ya1 = P[1]
          df_xy=df_xy.append({'x': xa1,'y': ya1,'path_loss': pl_d,}, ignore_index=True)
    return df_xy


def difference_of (df_r_1, df_r_2): #replaces empty Values with 113dB
    df_diff = pd.DataFrame(columns=['tx_beam','rx_beam','path_loss'])
    for tx_beam_r in range(0,64):			#TX
        for rx_beam_r in range(0,64):    	#RX

            if (df_r_1.path_loss[(df_r_1.tx_beam == rx_beam_r) & (df_r_1.rx_beam == tx_beam_r)].empty) == FALSE:
                if(df_r_1.path_loss[(df_r_1.tx_beam == rx_beam_r) & (df_r_1.rx_beam == tx_beam_r)].hasnans):
                    Value_df1 = 113
                else:
                    Value_df1 = int(df_r_1.path_loss[(df_r_1.tx_beam == rx_beam_r) & (df_r_1.rx_beam == tx_beam_r)])
            else: Value_df1 = 113
            #print (Value_df1)

            if (df_r_2.path_loss[(df_r_2.tx_beam == rx_beam_r) & (df_r_2.rx_beam == tx_beam_r)].empty) == FALSE:
                if(df_r_2.path_loss[(df_r_2.tx_beam == rx_beam_r) & (df_r_2.rx_beam == tx_beam_r)].hasnans):
                    Value_df2 = 113
                else:
                    Value_df2 = int(df_r_2.path_loss[(df_r_2.tx_beam == rx_beam_r) & (df_r_2.rx_beam == tx_beam_r)])
            else: Value_df2 = 113
            #print (Value_df2)

            path_loss_diff_r = Value_df1 - Value_df2

            df_diff=df_diff.append({'tx_beam': tx_beam_r,'rx_beam': rx_beam_r,'path_loss': path_loss_diff_r,}, ignore_index=True)
    #print("diff end, print solution:")
    #print (df_diff)
    return df_diff


def generate_sorted_df(df_g):
    df_1_sorted = pd.DataFrame(columns=['tx_beam','rx_beam','path_loss'])
    lengthy = len(df_g.index)
    for i in range(0,lengthy):

        pl_new = df_g.path_loss[i]
        tx_old = df_g.tx_beam[i]
        rx_old = df_g.rx_beam[i]

        tx_new, rx_new = sort_to_xy(tx_old, rx_old)
        df_1_sorted=df_1_sorted.append({'tx_beam': tx_new,'rx_beam': rx_new,'path_loss': pl_new,}, ignore_index=True)
        #print("Dataset: tx_old:"+str(tx_old)+ " tx_new:"+str(tx_new)+" rx_old:"+str(rx_old)+ " rx_new:"+str(rx_new)+" pl"+str(pl_new))
        #print(df_1_sorted)

    return df_1_sorted

#Search for the smallest path_loss to find LOS(Line of Sight)
def ds_getCalibration(dataFrame, gC_Canvas):
   pl = 115
   for i in range(0,len(dataFrame.index)):
       pl_temp = dataFrame.path_loss[i]
       if (pl_temp<pl):
           pl=pl_temp
           tx_off = dataFrame.tx_beam[i]-32
           rx_off = dataFrame.rx_beam[i]-32
           #print("Newest pl min found: "+str(pl)+"dB at TX:"+str(tx_off)+" and RX:"+str(rx_off))
           gC_Canvas.create_text(20,540,anchor=NW, text="TX Offset:"+str((tx_off*90/64))+"°", font=("Arial", 10))
           gC_Canvas.create_text(900,540,anchor=NW, text="RX Offset:"+str((rx_off*90/64))+"°", font=("Arial", 10))
       return tx_off, rx_off, pl





def ds_read_pathloss_div_from_csv (rx_beam_r, tx_beam_r, df_r_1, df_r_2): #replaces empty Values with 113dB

    if (df_r_1.path_loss[(df_r_1.tx_beam == rx_beam_r) & (df_r_1.rx_beam == tx_beam_r)].empty) == FALSE:
        if(df_r_1.path_loss[(df_r_1.tx_beam == rx_beam_r) & (df_r_1.rx_beam == tx_beam_r)].hasnans):
            Value_df1 = 113
        else:
            Value_df1 = int(df_r_1.path_loss[(df_r_1.tx_beam == rx_beam_r) & (df_r_1.rx_beam == tx_beam_r)])
    else: Value_df1 = 113
    #print (Value_df1)
         
    if (df_r_2.path_loss[(df_r_2.tx_beam == rx_beam_r) & (df_r_2.rx_beam == tx_beam_r)].empty) == FALSE:
        if(df_r_2.path_loss[(df_r_2.tx_beam == rx_beam_r) & (df_r_2.rx_beam == tx_beam_r)].hasnans):
            Value_df2 = 113
        else:
            Value_df2 = int(df_r_2.path_loss[(df_r_2.tx_beam == rx_beam_r) & (df_r_2.rx_beam == tx_beam_r)])
    else: Value_df2 = 113
    #print (Value_df2)


    path_loss_diff_r = Value_df1 - Value_df2

    return path_loss_diff_r


def ds_draw_intersection(tx_beam_d, rx_beam_d, diff_df_df2, canvas_d):
        tx_degree = (((63-rx_beam_d)/63)*90)-45 # Transform from 0-63 to -45 Grad upto + 45 degree
        tx_steigung = math.tan(math.radians(tx_degree)) #elevation as XY-Factor
        position_x_TX = 100
        position_y_TX = 500
        tx_x1 = position_x_TX
        tx_y1 = position_y_TX
        tx_x2 = position_x_TX+(800)
        tx_y2 = position_y_TX+(800*tx_steigung)
        TX_Strahl = line([tx_x1,tx_y1], [tx_x2,tx_y2]) #needed for intersection calculation
        
        rx_degree = ((((63-tx_beam_d)/63)*90)-45)
        rx_steigung = math.tan(math.radians(-rx_degree))
        position_x_RX = 900
        position_y_RX = 500
        rx_x1 = position_x_RX
        rx_y1 = position_y_RX
        rx_x2 = position_x_RX-(800)
        rx_y2 = position_y_RX+(800*rx_steigung)
        RX_Strahl = line([rx_x1,rx_y1], [rx_x2,rx_y2]) #needed for intersection calculation
        

        p_size = diff_df_df2/4
        p_size_to_print = diff_df_df2

        P = intersection(RX_Strahl, TX_Strahl)
        if bool(P) and P[0]>100 and P[0]<900: # and p_size>12:
          x1, y1 = (P[0] - p_size), (P[1] - p_size)
          x2, y2 = (P[0] + p_size), (P[1] + p_size)
          #if int(p_size_to_print)<83:
          canvas_d.create_oval(x1, y1, x2, y2, fill='grey90')
          canvas_d.create_text(P[0], P[1], text=p_size_to_print, font=("Arial", 8))
          xa1 = P[0]
          ya1 = P[1]
          return xa1,ya1
        return 0,0



def draw_circle(cx,cy,c_size, c_color, c_canvas):
    x1, y1 = (cx - c_size), (cy - c_size)
    x2, y2 = (cx + c_size), (cy + c_size)
    c_canvas.create_oval(x1, y1, x2, y2, fill=c_color)
    return 0




#-----------------------------------------------------------------------
#-- Top-View calculation
def ds_draw_top_view(canvas_d):

    canvas_d.create_text(10,440,anchor=NW, text="Node 1 - TX", font=("Arial", 16))
    canvas_d.create_text(910,440,anchor=NW, text="Node 2 - RX", font=("Arial", 16))

    for i in range(0,64):
        tx_degree_line = (((i/63)*90)-45)
        tx_steigung_line = math.tan(math.radians(tx_degree_line))
        position_x_TX = 100
        position_y_TX = 500
        tx_x1 = position_x_TX
        tx_y1 = position_y_TX
        tx_x2 = position_x_TX+(800)
        tx_y2 = position_y_TX+(800*tx_steigung_line)
        canvas_d.create_line(tx_x1,tx_y1,tx_x2,tx_y2,fill='grey90')
        position_x_RX = 900
        position_y_RX = 500
        rx_x1 = position_x_RX
        rx_y1 = position_y_RX
        rx_x2 = position_x_RX-(800)
        rx_y2 = position_y_RX+(800*tx_steigung_line)
        canvas_d.create_line(rx_x1,rx_y1,rx_x2,rx_y2,fill='grey90')
    return 0

#----------------------------------------------------------------------   
#-- XY-View calculation
def ds_draw_xy_view(canvas_d):    
    draw_offset_x = 100
    draw_offset_y = 100
    draw_scale = 1

    for i in range(0,64):
        x1= draw_offset_x+12*i*draw_scale
        y1= draw_offset_y+12*i*draw_scale
        canvas_d.create_line(x1,draw_offset_y,x1,draw_offset_y+63*12, fill='grey90')
        canvas_d.create_line(draw_offset_x,y1,draw_offset_x+63*12,y1, fill='grey90')

#----------------------------------------------------------------------
#-- XY-comparison-View calculation
def ds_draw_vgl_view(df_vgl, ds_canvas_vgl):

    listlength = len(df_vgl.index)
    #print("start")
    for i in range(0,listlength):
        if((str(df_vgl.path_loss[i])=='nan') or (df_vgl.path_loss[i]>120)):
                    #print("NaN-Alarm")
                    pass
        else:
            d_tx = df_vgl.tx_beam[i]
            d_rx = df_vgl.rx_beam[i]
            d_pl = df_vgl.path_loss[i]
            if (d_tx<32 and d_rx>31):
                ds_canvas_vgl.create_text(((d_tx+32)*15+20), (d_rx*15+20), text=(str(int(d_pl))), font=("Arial", 8), fill="black")
            if (d_tx>31 and d_rx>31):
                ds_canvas_vgl.create_text(((-d_tx+63)*15+20), (d_rx*15+20), text=(str(int(d_pl))), font=("Arial", 8), fill="black")
            if (d_tx<32 and d_rx<32):
                ds_canvas_vgl.create_text(((d_tx+32)*15+20), ((-d_rx+31)*15+20), text=(str(int(d_pl))), font=("Arial", 8), fill="black")
            if (d_tx>31 and d_rx<32):
                ds_canvas_vgl.create_text(((-d_tx+63)*15+20), ((-d_rx+31)*15+20), text=(str(int(d_pl))), font=("Arial", 8), fill="black")

    for i in range(0,64):
        ds_canvas_vgl.create_text(((i)*15+20), (10), text=(str(i)), font=("Arial", 12), fill="blue")
        ds_canvas_vgl.create_text(10, (((i)*15+20)), text=(str(i)), font=("Arial", 12), fill="blue")

def ds_draw_df3(df_vgl, ds_canvas_vgl):

    listlength = len(df_vgl.index)
    for i in range(0,listlength):
        if((str(df_vgl.path_loss[i])=='nan')or (df_vgl.path_loss[i]>100)):
                    #print("NaN-Alarm")
                    pass
        else:
            d_tx = df_vgl.tx_beam[i]
            d_rx = df_vgl.rx_beam[i]
            d_pl = df_vgl.path_loss[i]
            ds_canvas_vgl.create_text((d_tx*15+20), (d_rx*15+20), text=(str(int(d_pl))), font=("Arial", 8), fill="black")

    for i in range(0,64):
        ds_canvas_vgl.create_text(((i)*15+20), (10), text=(str(i)), font=("Arial", 12), fill="blue")
        ds_canvas_vgl.create_text(10, (((i)*15+20)), text=(str(i)), font=("Arial", 12), fill="blue")




#----------------------------------------------------------------------   
# Intersection helpers
# Inspired by : https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2): 
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False




def sort_to_xy(d_tx, d_rx): # Die Eingegebenen Daten müssen umsortiert werden zu: 0=-45°, 63=45°
    if (d_tx<32 and d_rx>31):
        tx_output=d_tx+32
        rx_outputz=d_rx
    if (d_tx>31 and d_rx>31):
        tx_output=-d_tx+63
        rx_outputz=d_rx
    if (d_tx<32 and d_rx<32):
        tx_output=d_tx+32
        rx_outputz=-d_rx+31
    if (d_tx>31 and d_rx<32):
        tx_output=-d_tx+63
        rx_outputz=-d_rx+31

    return tx_output,rx_outputz





def DBIS (df_dbis, canvas_d, dbis_eps, dbis_min_samples): # Variaton from https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py


    import numpy as np

    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler


    # #############################################################################

    if df_dbis.size>0: # needed, fit_transform whould crash if array is empty (no points to cluster)
        X = StandardScaler().fit_transform(df_dbis)
    else:
        return 0
    

    
    #dom_eps = 100
    dom_min_samples = 6

    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=dbis_eps, min_samples=dbis_min_samples).fit(df_dbis)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    number_of_groups = 0
    array_of_groups = pd.DataFrame(columns=['x','y','pl_sum'])
   
    for i in range(0,labels.size):      # Anzahl der Wolken finden
        if labels[i]>number_of_groups: 
            number_of_groups=labels[i]
    listofsolutions = []

    
    for i in range(0,number_of_groups+1):

        x_sum=0
        y_sum=0
        members_in_group=0

        for j in range(0,labels.size):  # For every point in a cloud
            if (labels[j]==i):
                x_sum = x_sum + df_dbis.x[j]
                y_sum = y_sum + df_dbis.y[j]
                members_in_group=members_in_group+1
                
        if members_in_group>1:
            x_position = x_sum/members_in_group-1
            y_position = y_sum/members_in_group-1
        else:
            x_position = 0
            y_position = 0

    
        listofsolutions.append([((x_position-100)/4),((y_position-500)/4)]) #absolute position in CM is calculated back.
        if members_in_group >= dbis_min_samples:
            draw_circle(x_position, y_position,10, 'yellow', canvas_d)
            canvas_d.create_text(x_position, y_position, text=str(i+1), font=("Arial", 20), fill="black")
            canvas_d.create_text(150, (60+15*i),anchor=NW, text=("Object found(eps="+str(dbis_eps)+",min_samples="+str(dbis_min_samples)+"): Cluster with "+str(members_in_group)+" Points at X:"+str(int(((x_position-100)/4)))+"cm Y:"+str(int(((y_position-500)/4)))+"cm (yellow circle)"), font=("Arial", 10))



    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    return listofsolutions

