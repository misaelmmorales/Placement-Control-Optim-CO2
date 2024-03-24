mrstModule add ad-core ad-props co2lab coarsegrid

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


%% Grid, petrophysical data, well, and initial state
% The volumetric grid model is created in a separate routine, which removes
% the shale layers from the Dunhil and Amundsen formations so that the
% resulting 3D grid only consists of the high permeability sand. The
% Johansen formation is partially delimited by sealing faults, and the
% routine therefore also identifies the parts of the vertical boundary
% assumed to be open, returned as a list of face indices. We then create a
% top-surface grid to represent the sealing caprock, and place a single
% injection well far downdip, close to the major sealing fault. The
% formation is assumed to contain pure brine in hydrostatic equilibrium.

%% Grid and open boundary faces
[G, rock, bcIx, ~, ~, bcIxVE] = makeJohansenVEgrid();

[Gt, ~, ~] = topSurfaceGrid(G);


%% Setup the well
num_wells = randi([1,3]);
well_locs = randsample(Gt.cells.indexMap, num_wells);

inj_rate  = 3 * mega * 1e3 / year / co2_rho;
max_rate  = 6.0 * mega * 1e3 / year / co2_rho;
max_bhp   = 5000 * psia;

W = [];
for i=1:num_wells   
    W = addWell(W, G, rock, well_locs(i), ...
                'name', ['Injector', int2str(i)], ...
                'sign', 1, ...
                'InnerProduct', 'ip_tpf', ...
                'type', 'rate', ...
                'val', inj_rate / num_wells, ...
                'lims', max_bhp, ...
                'comp_i', [0 1]);
end

%% Petrophysical data, well, and initial state
[Gt, G, transMult] = topSurfaceGrid(G);

rock2D             = averageRock(rock, Gt);
W2D                = convertwellsVE(W, G, Gt, rock2D);
initState.pressure = rhow * g(3) * Gt.cells.z;
initState.s        = repmat([1, 0], Gt.cells.num, 1);
initState.sGmax    = initState.s(:,2);

%% Fluid model
% The PVT behavior of the injected CO2 is assumed to be given by an
% equation state, whereas the brine is incompressible. We also include a
% capillary fringe model based on upscaled, sampled capillary pressure.
% Please consult the documentation for the makeVEFluid routine for a
% description of parameters you can use to set up the various fluid models
% implemented in MRST-co2lab.
invPc3D = @(pc) (1-srw) .* (pe./max(pc, pe)).^2 + srw;
kr3D    = @(s) max((s-src)./(1-src), 0).^2; % uses CO2 saturation
fluid   = makeVEFluid(Gt, rock, 'P-scaled table'             , ...
               'co2_mu_ref'  , muco2, ... %6e-5 * Pascal * second , ...
               'wat_mu_ref'  , muw, ... %8e-4 * Pascal * second , ...
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

%% Set up simulation schedule
% The simulation will consist of two periods: during the first 10 years,
% CO2 is injected as constant rate from the single injector. This period is
% simulated with a time step of 1 year. We then simulate a post-injection
% period of 500 years using time steps of 10 years.

% hydrostatic pressure conditions for open boundary faces
p_bc     = Gt.faces.z(bcIxVE) * rhow * g(3);
bc2D     = addBC([], bcIxVE, 'pressure', p_bc); 
bc2D.sat = repmat([1 0], numel(bcIxVE), 1);

% Setting up two copies of the well and boundary specifications. 
% Modifying the well in the second copy to have a zero flow rate.
schedule.control    = struct('W', W2D, 'bc', bc2D);
schedule.control(2) = struct('W', W2D, 'bc', bc2D);

for i=1:num_wells
    schedule.control(2).W(i).val = 0;
end

% Specifying length of simulation timesteps
schedule.step.val = [repmat(year/2,  20, 1); ...
                     repmat(10*year, 50, 1)];

% Specifying which control to use for each timestep.
% The first 20 timesteps will use control 1, the last 50
% timesteps will use control 2.
schedule.step.control = [ones(20,1)*1; ...
                         ones(50,1)*2];

%% Create and simulate model
model = CO2VEBlackOilTypeModel(Gt, rock2D, fluid);
[wellSol, states] = simulateScheduleAD(initState, model, schedule);
states = [{initState} states(:)'];

%% Animate the plume migration over the whole simulation period
figure

oG = generateCoarseGrid(Gt.parent, ones(Gt.parent.cells.num,1));
plotFaces(oG, 1:oG.faces.num,'FaceColor','none');
plotWell(Gt.parent, W,'FontSize',10);

view(-63, 50); axis tight; colorbar, clim([0 1-srw]); colormap(parula.^2);
hs     = [];
time   = cumsum([0; schedule.step.val])/year;
period = [1; schedule.step.control];
ptxt   = {'injection','migration'};

for i=1:numel(states)
    delete(hs)
    [h, h_max] = upscaledSat2height(states{i}.s(:,2), states{i}.sGmax, Gt, ...
                                    'pcWG', fluid.pcWG, ...
                                    'rhoW', fluid.rhoW, ...
                                    'rhoG', fluid.rhoG, ...
                                    'p', states{70}.pressure);
    sat = height2Sat(struct('h', h, 'h_max', h_max), Gt, fluid);
    title(sprintf('Time: %4d yrs (%s)', time(i),ptxt{period(i)}));
    ix = sat>0; if ~any(ix), continue; end
    hs = plotCellData(Gt.parent, sat, ix); drawnow
end

%% Trapping inventory
% The result is a more detailed inventory that accounts for six different categories of CO2:
%
% # Structural residual - CO2 residually trapped inside a structural trap
% # Residual - CO2 residually trapped outside any structural traps
% # Residual in plume - fraction of the CO2 plume outside any structural
%   traps that will be left behind as residually trapped droplets when the
%   plume migrates away from its current position
% # Structural plume - mobile CO2 volume that is currently contained within
%   a residual trap;  if the containing structure is breached, this volume
%   is free to migrate upward
% # Free plume - the fraction of the CO2 plume outside of structural traps
%   that is free to migrate upward and/or be displaced by imbibing brine.
% # Exited - volume of CO2 that has migrated out of the domain through its
%   lateral boundaries
%
% This model only has very small structural traps and residual trapping is
% therefore the main mechanism.
ta = trapAnalysis(Gt, false);
reports = makeReports(Gt, states, model.rock, model.fluid, ...
                      schedule, [srw, src], ta, []);

h1 = figure; plot(1); ax = get(h1, 'currentaxes');
plotTrappingDistribution(ax, reports, 'legend_location', 'northwest');

%% END