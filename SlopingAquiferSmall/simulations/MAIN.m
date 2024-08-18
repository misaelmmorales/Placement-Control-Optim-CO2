%% Main variables
set(0,'DefaultFigureWindowStyle','docked')
% Import MRST module
mrstModule add co2lab mimetic matlab_bgl
mrstModule add ad-core ad-props ad-blackoil mrst-gui
clear;clc

% Define global variables
dims = 64;

%% Make Grid
[G, Gt, ~, ~, bcIx, bcIxVE] = makeModel(32,1);
save('grids/G.mat', 'G')
save('grids/Gt.mat', 'Gt')

figure(1); clf; plotCellData(G, G.cells.centroids(:,3)); view(3); colormap jet; colorbar

%% Make Initial State
gravity on;  g = gravity;
rhow = 1000;
P0 = 4000*psia;
initState.pressure = repmat(P0, dims*dims, 1);
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

% Modify relative permeability curves and capillary pressure
srw = 0.27;
src = 0.20;
fluid.krW = @(s) fluid.krW(max((s-srw)./(1-srw), 0));
fluid.krG = @(s) fluid.krG(max((s-src)./(1-src), 0));
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

%% Generate Models
perm = load('data/perm_64x64.mat').perm';
poro = load('data/poro_64x64.mat').poro';
facies = load('data/facies_64x64.mat').facies';

timesteps  = [1010*year(), 10*year(), 0.5*year(), 50*year()];
time_arr = [repmat(timesteps(3)/year, timesteps(2)/timesteps(3), 1); 
            repmat(timesteps(4)/year, (timesteps(1)-timesteps(2))/timesteps(4), 1)];
save('data/time_arr.mat', 'time_arr')

total_inj  = (10 / (timesteps(2)/year) ); % 10 MT over 10 yrs = 1 MT/yr
min_inj    = 0.2; % in MT CO2
conversion = rhoc * (year/2) / 1e3 / mega;

%% Run Simulation
parfor i=0:1271
    [rock]            = gen_rock(i, perm, poro, facies);
    [W, WVE, wellIx]  = gen_wells(G, Gt, rock);
    [controls]        = gen_controls(timesteps, total_inj, min_inj, W, fluid);
    [schedule]        = gen_schedule(timesteps, W, bc, controls);
    [wellSol, states] = gen_ADsimulation(G, rock, fluid, initState, schedule);

    parsave(sprintf('states/states_%d', i), states);
    parsave(sprintf('controls/controls_%d', i), controls*conversion);
    parsave(sprintf('well_locs/well_locs_%d', i), wellIx);
    parsave(sprintf('rock/rock_%d', i), rock);
    parsave(sprintf('porevol/pv_%d', i), poreVolume(G,rock));
    fprintf('Simulation %i done\n', i)
end
disp('... All Done!');

%% END