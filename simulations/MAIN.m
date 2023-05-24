%% Main variables
clear; close all; clc
set(0,'DefaultFigureWindowStyle','docked')

% Import MRST module
mrstModule add SPE10 co2lab mrst-gui
mrstModule add ad-core ad-props ad-blackoil

% Define global variables
dims = 32;

%% Make Grid
nx=dims; ny=dims; nz=1; 
dx=1000*meter; dy=1000*meter; dz=100*meter;

% Make cartesian grid
G = cartGrid([nx ny nz], [dx dy dz]);
G = computeGeometry(G);

%% Make Rock
logperm = readmatrix('perm_realization.csv');
random = 0.2*randn([dims*dims,1]);

poro = 10.^((logperm-7)/10);
permx = (10.^(0.25+logperm+random))*milli*darcy;
permy = permx;
permz = 0.1*permx;
perm = [permx, permy, permz];
rock = makeRock(G, perm, poro);

permeability = convertTo(perm, milli*darcy);
save('permeability.mat', 'permeability')
save('porosity.mat', 'poro')

%% Make Initial State
gravity on;  g = gravity;
rhow = 1000; % density of brine corresponding to 94 degrees C and 300 bar
%initState.pressure = G.cells.centroids(:,3) * 400*psia; %G.cells.centroids=3
P0 = 4000*psia; 
initState.pressure = repmat(P0, 1024,1);
initState.s = repmat([1, 0], G.cells.num, 1);
initState.sGmax = initState.s(:,2);

%% Make Fluid
co2     = CO2props();             % load sampled tables of co2 fluid properties
p_ref   = 30 * mega * Pascal;     % choose reference pressure
t_ref   = 94 + 273.15;            % choose reference temperature, in Kelvin
rhoc    = co2.rho(p_ref, t_ref);  % co2 density at ref. press/temp
cf_co2  = co2.rhoDP(p_ref, t_ref) / rhoc; % co2 compressibility
cf_wat  = 0;                      % brine compressibility (zero)
cf_rock = 4.35e-5 / barsa;        % rock compressibility
muw     = 8e-4 * Pascal * second; % brine viscosity
muco2   = co2.mu(p_ref, t_ref) * Pascal * second; % co2 viscosity

% Use function 'initSimpleADIFluid' to make a simple fluid object
fluid = initSimpleADIFluid('phases', 'WG'           , ...
                           'mu'  , [muw, muco2]     , ...
                           'rho' , [rhow, rhoc]     , ...
                           'pRef', p_ref            , ...
                           'c'   , [cf_wat, cf_co2] , ...
                           'cR'  , cf_rock          , ...
                           'n'   , [2 2]);

% Modify relative permeability curves
srw = 0.27;
src = 0.20;
fluid.krW = @(s) fluid.krW(max((s-srw)./(1-srw), 0));
fluid.krG = @(s) fluid.krG(max((s-src)./(1-src), 0));

% Add capillary pressure
pe = 5 * kilo * Pascal;
pcWG = @(sw) pe * sw.^(-1/2);
fluid.pcWG = @(sg) pcWG(max((1-sg-srw)./(1-srw), 1e-5)); 

%% Make Boundary Conditions
bc = [];
vface_ind = (G.faces.normals(:,3) == 0);
bface_ind = (prod(G.faces.neighbors, 2) == 0);
bc_face_ix = find(vface_ind & bface_ind);
bc_cell_ix = sum(G.faces.neighbors(bc_face_ix, :), 2);
p_face_pressure = initState.pressure(bc_cell_ix);
bc = addBC(bc, bc_face_ix, 'pressure', p_face_pressure, 'sat', [1,0]);

%% Define Timesteps
total_time = 5*year;
timestep = rampupTimesteps(total_time, year/12, 0);
cum_time = convertTo(cumsum(timestep), year);
save('time_yr.mat', 'cum_time')

irate = sum(poreVolume(G, rock))/(total_time);

%% Run Simulations
N_realization = 2;

well_locations = cell(N_realization,1);
results = cell(1,N_realization);

parfor i=1:N_realization
    inj_loc = randi([4,28],[1,2]);
    W = gen_wells(G, rock, inj_loc, irate);
    [schedule, dT1] = gen_schedule(W, bc, timestep);
    [model, wellSol, states] = gen_simulation(G, rock, fluid, initState, schedule);
    well_locations{i} = inj_loc;
    results{i} = states;
end
save('results.mat', 'results')

%% Collect Results
well_locations = cell2mat(well_locations);
save('well_locations.mat', 'well_locations')

pressure = zeros(N_realization,dims*dims,length(timestep));
saturation = zeros(N_realization,dims*dims,length(timestep));
bhp = zeros(N_realization,length(timestep));
for i=1:N_realization
    for j=1:length(timestep)
        pressure(i,:,j) = convertTo(results{1,i}{j,1}.pressure, psia);
        saturation(i,:,j) = results{1,i}{j,1}.s(:,2);
        bhp(i,j) = convertTo(results{1,i}{j,1}.wellSol.bhp, psia);
    end
end
save('pressure.mat','pressure')
save('saturation.mat','saturation')
save('bhp.mat', 'bhp')

%% Plots
%{
figure; plotToolbar(G, results{1}); colormap jet

figure; 
plotCellData(G, rock.poro); 
plotWell(G, W, 'color', 'k');
title('Porosity & Well Locations'); xlabel('X [m]'); ylabel('Y [m]')
view(-50,85); colormap jet; h=colorbar; h.Label.String = 'Porosity [v/v]';

figure; plotCellData(G, states{1}.s(:,1)); colormap jet; view(-50,85)
figure; plotCellData(G, states{end}.s(:,1)); colormap jet; view(-50,85)

bhp_inj = zeros(length(timestep),1);
for i=1:length(timestep)
    bhp_inj(i) = convertTo(states{i}.wellSol(1).bhp, psia);
end
figure; plot(cum_time, bhp_inj, 'DisplayName','Injector');
legend; title('BHP vs. Time'); xlabel('Time [years]'); ylabel('BHP [psia]')
%}
%% END