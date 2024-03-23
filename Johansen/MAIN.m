clear;clc;close all
%% Main variables
set(0,'DefaultFigureWindowStyle','docked')

proj_dir = 'E:/Placement-Control-Optim-CO2';
mdir = 'C:/Users/Misael Morales/OneDrive - The University of Texas at Austin/Documents/MATLAB/mrst-2022b';
chdir(mdir);
startup;
chdir(proj_dir);

% Import MRST module
mrstModule add spe10 coarsegrid co2lab ad-core ad-props ad-blackoil mrst-gui

%% Make Johansen Grid

dpath = getDatasetPath('johansen');
sector = fullfile(dpath, 'NPD5');
filename = [sector, '.grdecl'];

grdecl = readGRDECL(filename);  clear filename
grdecl.ACTNUM = grdecl.ACTNUM;
G = processGRDECL(grdecl);
G = computeGeometry(G);

nx = G.cartDims(1);
ny = G.cartDims(2);
nz = G.cartDims(3);

%% Ground Truth model

p = reshape(load([sector, '_Porosity.txt'])', prod(G.cartDims), []);
poro = p(G.cells.indexMap); clear p

K = reshape(load([sector, '_Permeability.txt']')', prod(G.cartDims), []);
perm = bsxfun(@times, [1 1 0.1], K(G.cells.indexMap)).*milli*darcy; clear K;
rock = makeRock(G, perm, poro);

w = load([sector, '_Well.txt']);

%% Make Initial State
gravity on;
g = gravity;
rhow = 1000;
initState.pressure = rhow * g(3) * G.cells.centroids(:,3);
initState.s = repmat([1, 0], G.cells.num, 1);
initState.sGmax = initState.s(:,2);

%% Make Fluid
co2     = CO2props(); % load sampled tables of co2 fluid properties
p_ref   = 15 * mega * Pascal; % choose reference pressure
t_ref   = 70 + 273.15; % choose reference temperature, in Kelvin
rhoc    = co2.rho(p_ref, t_ref); % co2 density at ref. press/temp
cf_co2  = co2.rhoDP(p_ref, t_ref) / rhoc; % co2 compressibility
cf_wat  = 0; % brine compressibility (zero)
cf_rock = 4.35e-5 / barsa; % rock compressibility
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

% Change relperm curves
srw = 0.27;
src = 0.20;
fluid.krW = @(s) fluid.krW(max((s-srw)./(1-srw), 0));
fluid.krG = @(s) fluid.krG(max((s-src)./(1-src), 0));

% Add capillary pressure curve
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

%% Define timesteps
dT1 = rampupTimesteps(10*year,year/2,0);  
dT2 = rampupTimesteps(100*year, 5*year, 0);
dT = [dT1; dT2];
dTinj = sum(dT1)/year;

%% Ground Truth simulation
irate = 50 * mega * 1e3 / year / rhoc;
maxrate = 10 * mega * 1e3 / year / rhoc;

W = verticalWell([], G, rock,  w(1), w(2), w(3):w(4),  ...
                 'InnerProduct', 'ip_tpf', ...
                 'sign', 1,...
                 'Radius', 0.1, ...
                 'name', 'I',...
                 'comp_i', [0,1], ...
                 'type', 'rate', ...
                 'Val', irate);

schedule.control    = struct('W', W, 'bc', bc);
schedule.control(2) = struct('W', W, 'bc', bc);
schedule.control(2).W.val = 0;

schedule.step.val = [dT1; dT2];
schedule.step.control = [ones(numel(dT1),1); ones(numel(dT2),1)*2];

model = TwoPhaseWaterGasModel(G, rock, fluid, 0, 0);
[wellSol, states] = simulateScheduleAD(initState, model, schedule);

%%
figure; plotToolbar(G,states); plotWell(G,W); colormap jet; colorbar; view(-145,60)

















%% Generate Models & Run Simulation
N = 1000;
M = size(total_time,1);

perm = load('perm_64x64.mat').perm';
poro = load('poro_64x64.mat').poro';
facies = load('facies_64x64.mat').facies';

parfor i=1:N
    nwells                   = randi([1,3], 1);
    well_loc                 = randi([16,48], nwells, 2);
    rock                     = gen_rock(i, perm, poro, facies);
    W                        = gen_wells(G, rock, well_loc);
    [schedule, dT1]          = gen_schedule(W, bc, timestep1);
    [model, wellSol, states] = gen_simulation(G, rock, fluid, initState, schedule);

    wname = sprintf('wells/wells%d', i);   parsave(wname, well_loc)
    rname = sprintf('states/states%d', i); parsave(rname, states)
    fprintf('Simulation %i done\n', i)
end

%% END