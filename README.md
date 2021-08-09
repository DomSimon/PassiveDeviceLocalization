#Device-free passive localization using Millimeter Wave Radios.

This Code was written during my Master Thesis about passive localization of objects.
The Thesis uses the data of two phased antenna arrays to measure the signal loss at different directions of transmitted and received beams. One calibration session and one detection session is needed to generate proper results on a setup with two Terragraph radios 2 m apart. Path loss differences can then be computed and combined with intersection points of the transmitted and received beam angles. With this conjunction between intersection points and path loss the algorithm generates a two dimensional map with path loss differences at beam intersecting positions. In difference to many other algorithms this algorithm needs only one measurement session without any human interaction as calibration. Then objects can be placed in the detection area. After a new measurement session the algorithm uses the difference in path loss between the sessions to build clusters as representation of the position of objects in the search area. The results show, that multiple objects can be placed and will be found in a two square meter sized environment with a 85.9% detection rate and only 6.51 cm localization error. The algorithm described in this thesis describes a good way for passive localization without the need for labour intense calibrations.


<img width="1248" alt="tg" src="https://user-images.githubusercontent.com/63147491/128741781-95204cce-d69f-4337-9bcb-223b91a1390b.png">

### Specifications

29 Measurement sessions are available. 
The first two sessions have no obstacles placed. 

Ground truth is provided as positions.csv for 
every session and was collected manually. 

More details can be found in the thesis
"Device-free passive localization using Millimeter Wave Radios".
accuracy.py was kindly provided by Dr. Anatolij Zubow (TU Berlin).


### Localization can be started via Terminal:

If you want to use Session 2 as calibration and 
localize objects in Session 5 type:

```python3 tg_localization.py 2 5```



Way of finding the best Parameters via Terminal:
----------------------------------------------------------------
Adjust tg_findparam.py for a fitting purpose.
If nothing is changed, the script calculates alls 25536 combinations.
Averages are calculated and saved to results_iteration_sorted.csv
This table can then be sorted and searched by the user 
accordingly to the users priorities.

```python3 tg_findparam.py```

(Good priorities can be  only use lm_Pfa_avg=0 combinations, 
Chose combination with highest lm_Pd_avg)

