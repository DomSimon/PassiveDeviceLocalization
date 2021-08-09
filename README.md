Welcome to the programming code of Device-free passive localization using Millimeter Wave Radios By Dominic Simon
----------------------------------------------------------------

29 Measurement sessions are available. 
The first two sessions have no obstacles placed. 

Ground truth is provided as positions.csv for 
every session and was collected manually. 

More details can be found in the thesis
"Device-free passive localization using Millimeter Wave Radios".
accuracy.py was kindly provided by Dr. Anatolij Zubow (TU Berlin).


Localization can be started via Terminal:
----------------------------------------------------------------
If you want to use Session 2 as calibration and 
localize objects in Session 5 type:

python3 tg_localization.py 2 5



Way of finding the best Parameters via Terminal:
----------------------------------------------------------------
Adjust tg_findparam.py for a fitting purpose.
If nothing is changed, the script calculates alls 25536 combinations.
Averages are calculated and saved to results_iteration_sorted.csv
This table can then be sorted and searched by the user 
accordingly to the users priorities.

python3 tg_findparam.py

(Good priorities can be  only use lm_Pfa_avg=0 combinations, 
Chose combination with highest lm_Pd_avg)

