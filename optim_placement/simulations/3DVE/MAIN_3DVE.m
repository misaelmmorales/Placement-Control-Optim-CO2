%% Main variables
clear; close all; clc
set(0,'DefaultFigureWindowStyle','docked')

% Import MRST module
mrstModule add SPE10 co2lab mrst-gui coarsegrid
mrstModule add ad-core ad-props ad-blackoil

%% Fluid parameters
gravity reset on;
g       = gravity;
rhow    = 1000;                                 % water density (kg/m^3)
co2     = CO2props();                           % CO2 property functions
p_ref   = 30 *mega*Pascal;                      % reference pressure
t_ref   = 94+273.15;                            % reference temperature
co2_rho = co2.rho(p_ref, t_ref);                % CO2 density
co2_c   = co2.rhoDP(p_ref, t_ref) / co2_rho;    % CO2 compressibility
wat_c   = 0;                                    % water compressibility
c_rock  = 4.35e-5 / barsa;                      % rock compressibility
srw     = 0.27;                                 % residual water
src     = 0.20;                                 % residual CO2
pe      = 5 * kilo * Pascal;                    % capillary entry pressure
muw     = 8e-4 * Pascal * second;               % brine viscosity
muco2   = co2.mu(p_ref, t_ref) * Pascal * second; % co2 viscosity

%% Grid, petrophysical data, well, and initial state [1]
% Grid and open boundary faces
[G, rock, bcIx, ~, ~, bcIxVE] = makeJohansenVEgrid();

% Setup the well
total_time = 500*year;
inj_rate   = sum(poreVolume(G,rock))/total_time; %3.5*mega*1e3/year/co2_rho

inj_idx = randi([32,64],[1,2]);

wc_global = false(G.cartDims); 
wc_global(inj_idx(1), inj_idx(2), 6:10) = true;
wc = find(wc_global(G.cells.indexMap));
W  = addWell([], G, rock, wc, 'name', 'Injector',...
              'type', 'rate', ...  % inject at constant rate
              'val', inj_rate, ... % volumetric injection rate
              'comp_i', [0 1]);    % inject CO2, not water    

% Top surface grid, petrophysical data, well, and initial state
[Gt, G, transMult] = topSurfaceGrid(G);
rock2D             = averageRock(rock, Gt);
W2D                = convertwellsVE(W, G, Gt, rock2D);
initState.pressure = rhow * g(3) * Gt.cells.z;
initState.s        = repmat([1, 0], Gt.cells.num, 1);
initState.sGmax    = initState.s(:,2);

% To avoid plotting artifacts when visualizing the volumetric and the
% top-surface grids, we shift the volumetric grid 100 meters down,
GG = G; 
GG.nodes.coords(:,3) = GG.nodes.coords(:,3) + 100;

figure(1);
plotGrid(GG, 'facecolor', [1 1 .7]); 
plotGrid(Gt, 'facecolor', [.4 .5 1]);
[~,ht]=plotWell(G,W); set(ht,'FontSize',10, 'BackgroundColor',[.8 .8 .8]);
view(-70,30); clear GG;

%% Fluid model [2]
invPc3D = @(pc) (1-srw) .* (pe./max(pc, pe)).^2 + srw;
kr3D    = @(s) max((s-src)./(1-src), 0).^2; % uses CO2 saturation
fluid   = makeVEFluid(Gt, rock, 'P-scaled table'             , ...
               'co2_mu_ref'  , muco2, ...%6e-5*Pascal*second , ...
               'wat_mu_ref'  , muw, ...%8e-4*Pascal*second , ...
               'co2_rho_ref' , co2_rho                , ...
               'wat_rho_ref' , rhow                   , ...
               'co2_rho_pvt' , [co2_c, p_ref]         , ...
               'wat_rho_pvt' , [wat_c, p_ref]         , ...
               'residual'    , [srw, src]             , ...
               'pvMult_p_ref', p_ref                  , ...
               'pvMult_fac'  , c_rock                 , ...
               'invPc3D'     , invPc3D                , ...
               'kr3D'        , kr3D                   , ...
               'transMult'   , transMult);

%% Set up simulation schedule [3]
% hydrostatic pressure conditions for open boundary faces
p_bc     = Gt.faces.z(bcIxVE) * rhow * g(3);
bc2D     = addBC([], bcIxVE, 'pressure', p_bc); 
bc2D.sat = repmat([1 0], numel(bcIxVE), 1);

% Setting up two copies of the well and boundary specifications. 
% Modifying the well in the second copy to have a zero flow rate.
schedule.control    = struct('W', W2D, 'bc', bc2D);
schedule.control(2) = struct('W', W2D, 'bc', bc2D);
schedule.control(2).W.val = 0;

% Specifying length of simulation timesteps
schedule.step.val = [repmat(year,    50, 1); ...
                     repmat(10*year, 45, 1)];

% Specifying which control to use for each timestep.
% The first 100 timesteps will use control 1, 
% the last 100 timesteps will use control 2.
schedule.step.control = [ones(50, 1); ...
                         ones(45, 1) * 2];

%% Create and simulate model
model = CO2VEBlackOilTypeModel(Gt, rock2D, fluid);
[wellSol, states] = simulateScheduleAD(initState, model, schedule);
states = [{initState} states(:)'];

%% Plot
figure;
plotCellData(Gt, convertTo(states{1,end}.pressure, psia)); colormap jet;
h=colorbar; h.Label.String='Pressure [psia]'; view(-70,30)
xlabel('X'); ylabel('Y'); zlabel('Z'); title('Final Pressure State')

figure;
plotCellData(Gt, states{1,end}.s(:,2)); colormap jet;
h=colorbar; h.Label.String='CO2 Saturation [v/v]'; view(-70,30)
xlabel('X'); ylabel('Y'); zlabel('Z'); title('Final CO2 Saturation State')

%% Notes
% [1] The volumetric grid model is created in a separate routine, which 
% removes the shale layers from the Dunhil and Amundsen formations so that 
% the resulting 3D grid only consists of the high permeability sand. The
% Johansen formation is partially delimited by sealing faults, and the
% routine therefore also identifies the parts of the vertical boundary
% assumed to be open, returned as a list of face indices. We then create a
% top-surface grid to represent the sealing caprock, and place a single
% injection well far downdip, close to the major sealing fault. The
% formation is assumed to contain pure brine in hydrostatic equilibrium.

% [2] The PVT behavior of the injected CO2 is assumed to be given by an
% equation state, whereas the brine is incompressible. We also include a
% capillary fringe model based on upscaled, sampled capillary pressure.
% Please consult the documentation for the makeVEFluid routine for a
% description of parameters you can use to set up the various fluid models
% implemented in MRST-co2lab.

% [3] The simulation will consist of two periods: during the first 
% 50 years, CO2 is injected as constant rate from the single injector. 
% This period is simulated with a time step of 1 year. We then simulate a 
% post-injection period of 450 years using time steps of 10 years.

