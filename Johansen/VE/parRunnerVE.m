%% Main variables
proj_dir = pwd;
mdir = 'C:/Users/Misael Morales/OneDrive - The University of Texas at Austin/Documents/MATLAB/mrst-2022b';
%mdir = 'C:/Users/mmm6558/OneDrive - The University of Texas at Austin/Documents/MATLAB/mrst-2022b';
chdir(mdir); startup; chdir(proj_dir);
clear; clc; close all

set(0,'DefaultFigureWindowStyle','docked')
mrstModule add ad-core ad-props ad-blackoil co2lab coarsegrid mrst-gui linearsolvers
gravity on;

%% Grid
[G, ~, bcIx, Gt, ~, bcIxVE] = makeJohansenVEgrid();
[Gt, G, transMult] = topSurfaceGrid(G);

%% Initial State
g = gravity;
rhow = 1000;
initState.pressure = rhow * g(3) * Gt.cells.z;
initState.s = repmat([1, 0], Gt.cells.num, 1);
initState.sGmax = initState.s(:,2);

%% BC
% hydrostatic pressure conditions for open boundary faces
p_bc     = Gt.faces.z(bcIxVE) * rhow * g(3);
bc2D     = addBC([], bcIxVE, 'pressure', p_bc); 
bc2D.sat = repmat([1 0], numel(bcIxVE), 1);

%% VE Runner
n_realizations = 4; %318*4

parfor i=1:n_realizations
    [rock, rock2D]   = gen_rock_VE(i-1,G,Gt);
    [W, W2D, x, y]   = gen_wells_VE(G,Gt,rock,rock2D);
    [fluid_VE]       = gen_fluid_VE(rock,Gt,transMult);
    [schedule]       = gen_schedule_VE(W2D,bc2D,fluid_VE);
    [wellSol,states] = gen_simulation_VE(Gt,rock2D,fluid_VE,initState,schedule);
    parsave(sprintf('data_100_100_11/states_VE/states_%d', i-1), states)
    parsave(sprintf('data_100_100_11/well_coords_VE/well_coords_%d', i-1), struct('X',x,'Y',y));    
    fprintf('Simulation %i done\n', i)
end

%% END