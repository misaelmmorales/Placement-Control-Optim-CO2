%% Main variables
proj_dir = pwd;
mdir = 'C:/Users/Misael Morales/OneDrive - The University of Texas at Austin/Documents/MATLAB/mrst-2022b';
chdir(mdir); startup; chdir(proj_dir);
clear; clc; close all

set(0,'DefaultFigureWindowStyle','docked')
mrstModule add ad-core ad-props co2lab coarsegrid mrst-gui

%% Initialize Grid, Rock, Fluid, Initial State, Boundary Conditions
[G, rock, bcIx, ~, ~, ~, bcIxVE, grdecl] = make_Johansen(41);

gravity reset on;
g       = gravity;
co2     = CO2props();                             % CO2 property functions
p_ref   = 30 *mega*Pascal;                        % reference pressure
t_ref   = 94+273.15;                              % reference temperature
rhow    = 1000;                                   % water density (kg/m^3)
rhoc    = co2.rho(p_ref, t_ref);                  % CO2 density
c_co2   = co2.rhoDP(p_ref, t_ref) / rhoc;         % CO2 compressibility
c_water = 0;                                      % water compressibility
c_rock  = 4.35e-5 / barsa;                        % rock compressibility
srw     = 0.27;                                   % residual water
src     = 0.20;                                   % residual CO2
pe      = 5 * kilo * Pascal;                      % capillary entry pressure
muw     = 8e-4 * Pascal * second;                 % brine viscosity
muco2   = co2.mu(p_ref, t_ref) * Pascal * second; % co2 viscosity

%% Well(s)
inj_rate  = 3 * mega * 1e3 / year / rhoc;
max_bhp   = 10000 * psia;

num_wells = randi([1,3]);

increment = 8072;
actnum = reshape(grdecl.ACTNUM, G.cartDims);
actnum_l1 = actnum(:,:,1);
well_loc_l1 = randsample(find(actnum_l1(:)), num_wells);

well_locs = zeros(5,num_wells);
for i=1:num_wells
    well_locs(:,i) = (well_loc_l1(i) + (0:4)*increment)';
end

W   = []; 
W   = make_wells(W, G, rock, well_locs, inj_rate, []);

%% Vertical Equilibrium simulation
[Gt, ~, transMult] = topSurfaceGrid(G);
rock2D = averageRock(rock, Gt);

W2D = convertwellsVE(W, G, Gt, rock2D);

VE_initState.pressure = rhow*g(3)*Gt.cells.z;
VE_initState.s        = repmat([1,0], Gt.cells.num, 1);
VE_initState.sGmax    = VE_initState.s(:,2);

invPc3D  = @(pc) (1-srw) .* (pe./max(pc,pe)).^2 + srw; 
kr3D     = @(s) max((s-src)./(1-src), 0).^2;
VE_fluid = makeVEFluid(Gt, rock2D, 'P-scaled table'     , ...
                       'co2_mu_ref'  , muco2            , ...
                       'wat_mu_ref'  , muw              , ...
                       'co2_rho_ref' , rhoc             , ...
                       'wat_rho_ref' , rhow             , ...
                       'co2_rho_pvt' , [c_co2, p_ref]   , ...
                       'wat_rho_pvt' , [c_water, p_ref] , ...
                       'residual'    , [srw, src]       , ...
                       'pvMult_p_ref', p_ref            , ...
                       'pvMult_fac'  , c_rock           , ...
                       'invPc3D'     , invPc3D          , ...
                       'kr3D'        , kr3D             , ...
                       'transMult'   , transMult);

bc2D     = addBC([], bcIxVE, 'pressure', Gt.faces.z(bcIxVE) * rhow * g(3));
bc2D.sat = repmat([1,0], numel(bcIxVE), 1);

% schedule
min_rate  = 0.5             * mega * 1e3 / year / rhoc;
well_rate = 10 * rand(1,20) * mega * 1e3 / year / rhoc;
well_rate(well_rate<min_rate) = 0;
well_rate = well_rate * (50 / (rhoc*sum(well_rate*year/2)/mega/1e3));

for i=1:20
    VE_schedule.control(i) = struct('W', W2D, 'bc', bc2D);
end
VE_schedule.control(21) = struct('W', W2D, 'bc', bc2D);

for i=1:num_wells
    VE_schedule.control(21).W(i).val = 0;
    for k=1:20
        VE_schedule.control(k).W(i).val = well_rate(k);
    end
end

VE_schedule.step.val     = [repmat(year/2,20,1); repmat(50*year,10,1)];
VE_schedule.step.control = [linspace(1,20,20)';  ones(10,1)*21];

VE_model = CO2VEBlackOilTypeModel(Gt, rock2D, VE_fluid);
[VE_wellSol, VE_states] = simulateScheduleAD(VE_initState, VE_model, VE_schedule);
VE_states = [{VE_initState} VE_states(:)'];

figure
oG = generateCoarseGrid(Gt.parent, ones(Gt.parent.cells.num,1));
plotFaces(oG, 1:oG.faces.num,'FaceColor','none');
plotWell(Gt.parent, W,'FontSize',10);
view(-63, 50); axis tight; colorbar, clim([0 1-srw]); colormap(parula.^2);
hs     = [];
time   = cumsum([0; VE_schedule.step.val])/year;
period = [1; VE_schedule.step.control];
for i=1:numel(VE_states)
    delete(hs)
    [h, h_max] = upscaledSat2height(VE_states{i}.s(:,2), VE_states{i}.sGmax, Gt, ...
                                    'pcWG', VE_fluid.pcWG, ...
                                    'rhoW', VE_fluid.rhoW, ...
                                    'rhoG', VE_fluid.rhoG, ...
                                    'p', VE_states{end}.pressure);
    sat = height2Sat(struct('h', h, 'h_max', h_max), Gt, VE_fluid);
    title(sprintf('Time: %4d yrs', time(i)));
    hs = plotCellData(Gt.parent, sat); drawnow
end

ta = trapAnalysis(Gt, false);
reports = makeReports(Gt, VE_states, VE_model.rock, VE_model.fluid, VE_schedule, ...
                        [srw, src], ta, []);
h1 = figure; plot(1); ax = get(h1, 'currentaxes');
plotTrappingDistribution(ax, reports, 'legend_location', 'northwest');