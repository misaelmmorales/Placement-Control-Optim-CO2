ptf %
# run.dat
# ----------------------------SOLUTION TYPE----------------------------
sol
1	-1	
# -----------------------HISTORY OUTPUT REQUESTS-----------------------
node
18
1301 24710 65546 65556 65566 65576 66056 66066 66076 66086 66566 66576 66586 66596 67076 67086 67096 67106 
hist
days   100000  30.
mpa
deg
sco
co2m
end
#zfl
#cflz
#end
cont
tec 10000 1.e9
geom
material
perm
por
liquid
pressure
temperature
co2
end
# ------------------------------GENERATORS------------------------------
# -----------------------------PERMEABILITY-----------------------------
# 1: reservoir; 2: caprock; 3: aquifer
zone
file
/scratch/er/bailianchen/azmi/Example_3D/mesh/mesh_material.zone
perm
1	0	0	1e-16	1e-16	1e-16	
-11	0	0	%perm1% %perm1% %perm1%
-12	0	0	%perm2% %perm2% %perm2%
-13	0	0	%perm3% %perm3% %perm3%
26531 49940  2601 %perm4% %perm4% %perm4%

# -------------------------MATERIAL PARAMETERS-------------------------
rock
1    0   0   2563    1010    0.15
-11  0   0   2563    1010    0.15
-12  0   0   2563    1010    0.01
-13  0   0   2563    1010    0.15
26531 49940  2601 2563 1010 0.15 

# --------------------------ROCK CONDUCTIVITY--------------------------
cond
1	0	0	1.0	1.0	1.0	

# ------------------------RELATIVE PERMEABILITY------------------------
rlp
17 0 1 1 0 1 1 0 0 1 1 1 0 1 0 

1	0	0	1

# -----------------------TIME STEPPING PARAMETERS-----------------------
time
0.001	 720 	20000	 1	0.0	0.0	0.0	
30	-1.5	1.0	1	30.0	
60	-1.5	1.0	1	30.0	
90	-1.5	1.0	1	30.0	
120	-1.5	1.0	1	30.0	
150	-1.5	1.0	1	30.0	
180	-1.5	1.0	1	30.0	
210	-1.5	1.0	1	30.0	
240	-1.5	1.0	1	30.0	
270	-1.5	1.0	1	30.0	
300	-1.5	1.0	1	30.0
330	-1.5	1.0	1	30.0
360	-1.5	1.0	1	30.0	
390	-1.5	1.0	1	30.0	
420	-1.5	1.0	1	30.0	
450	-1.5	1.0	1	30.0	
480	-1.5	1.0	1	30.0	
510	-1.5	1.0	1	30.0
540	-1.5	1.0	1	30.0
570	-1.5	1.0	1	30.0
600	-1.5	1.0	1	30.0
630	-1.5	1.0	1	30.0
660	-1.5	1.0	1	30.0
690	-1.5	1.0	1	30.0
720	-1.5	1.0	1	30.0	

# --------------------SIMULATION CONTROL PARAMETERS--------------------
ctrl
-100	1e-06	100	200	gmre	
1	0	0	3	

1.0	3.0	1.0	
100	1.5	1e-07	24.35	
0	1	
# --------------------------SOLVER PARAMETERS--------------------------
iter
1e-06	1e-06	0.001	-0.01	1.15	
0	0	0	5	1e+11	
# ------------------------------CO2 MODULE------------------------------
zone
file
/scratch/er/bailianchen/azmi/Example_3D/mesh/mesh_outside.zone
# Set outflow boundary on four sides
flow
-3      0       0       0.0     -50.    -0.1
-4      0       0       0.0     -50.    -0.1
-5      0       0       0.0     -50.    -0.1
-6      0       0       0.0     -50.    -0.1

perm
1301 24710 2601 1e-13 1e-13 1e-10 # Set vertical perms high in well

carb
3
co2flow
-3      0       0       0.0     -50.    -0.1    2
-4      0       0       0.0     -50.    -0.1    2
-5      0       0       0.0     -50.    -0.1    2
-6      0       0       0.0     -50.    -0.1    2
1301    1301    1     -%q_co2%	-50	0	6  # from_node to_node cycle_by_n_nodes injection_rate(kg/s) injection_Temp(C) ignored mode_identifier

endcarb
stop
