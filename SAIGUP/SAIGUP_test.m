clear;clc;close all

mrstModule add SPE10 coarsegrid upscaling co2lab mrst-gui
mrstModule add ad-core ad-props ad-blackoil

grdecl = fullfile(getDatasetPath('SAIGUP'), 'SAIGUP.GRDECL');
grdecl = readGRDECL(grdecl);
usys   = getUnitSystem('METRIC');
grdecl = convertInputUnits(grdecl, usys);

actnum        = grdecl.ACTNUM;
grdecl.ACTNUM = ones(prod(grdecl.cartDims),1);

grdecl.ACTNUM = actnum; %clear actnum;
G = processGRDECL(grdecl);
G = computeGeometry(G);

rock = grdecl2Rock(grdecl, G.cells.indexMap);

%{
if 
    perm1 = 9600*1 double
then
    perm = perm1(find(actnum==1))
%}

%% Make Fluid
gravity on;  g = gravity;
rhow = 1000;
P0 = 4000*psia;
initState.pressure = repmat(P0, G.cells.num, 1);
initState.s = repmat([1, 0], G.cells.num, 1);
initState.sGmax = initState.s(:,2);

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
timestep1 = rampupTimesteps(80*year, 2*year, 0);
total_time = timestep1;

%% Simulation

R_inj = 5 * 556.2 * 1000 * meter^3 / year;
W = [];
W = verticalWell(W, G, rock, 20, 60, 10:20, ...
            'Type', 'rate', 'Val', R_inj, 'InnerProduct', 'ip_tpf', ...
            'Comp_i', [0,1], 'name', 'Inj');

schedule.control = struct('W', W, 'bc', bc);
schedule.step.val = timestep1;
schedule.step.control = ones(numel(timestep1),1);

model  = TwoPhaseWaterGasModel(G, rock, fluid);
[wellSol, states] = simulateScheduleAD(initState, model, schedule, ...
                                  'NonLinearSolver', NonLinearSolver());

%% END