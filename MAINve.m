%% Main variables
proj_dir = pwd;
mdir = 'C:/Users/Misael Morales/OneDrive - The University of Texas at Austin/Documents/MATLAB/mrst-2022b';
chdir(mdir); startup; chdir(proj_dir);
clear; clc; close all

set(0,'DefaultFigureWindowStyle','docked')
mrstModule add ad-core ad-props co2lab coarsegrid mrst-gui

%% Grid, Rock, BCs
grdecl = readGRDECL([fullfile(mrstPath('co2lab'),'data','johansen','NPD5'),'.grdecl']);

% load permeability and porosity
r = load(sprintf('data_100_100_11/rock/rock_%d.mat', i));
p = r.poro(:);
K = 10.^r.perm(:);

% Construct grid structure.
G = processGRDECL(grdecl);
G = computeGeometry(G(1));
[Gt, G, transMult] = topSurfaceGrid(G);

% Construct structure with petrophyiscal data.
rock.perm = bsxfun(@times, [1 1 0.1], K(G.cells.indexMap)).*milli*darcy;
rock.poro = p(G.cells.indexMap);
rock2D    = averageRock(rock, Gt);
clear p K;

% FIND PRESSURE BOUNDARY
% Setting boundary conditions is unfortunately a manual process and may
% require some fiddling with indices, as shown in the code below. Here, we
% identify the part of the outer boundary that is open, i.e., not in
% contact with one of the shales (Dunhil or Amundsen).

% boundary 3D
nx = G.cartDims(1); ny=G.cartDims(2); nz=G.cartDims(3);
ix1 = searchForBoundaryFaces(G, 'BACK', 1:nx-6, 1:4, 1:nz);
ix2 = searchForBoundaryFaces(G, 'LEFT', 1:20,   1:ny, 1:nz);
ix3 = searchForBoundaryFaces(G, 'RIGHT', 1:nx, ny-10:ny, 1:nz);
ix4 = searchForBoundaryFaces(G, 'FRONT', 1:nx/2-8, ny/2:ny, 1:nz);
bcIx = [ix1; ix2; ix3; ix4];

% boundary 2D
nx = Gt.cartDims(1); ny=Gt.cartDims(2);
ix1 = searchForBoundaryFaces(Gt, 'BACK',  1:nx-6, 1:4, []);
ix2 = searchForBoundaryFaces(Gt, 'LEFT',  1:20, 1:ny,  []);
ix3 = searchForBoundaryFaces(Gt, 'RIGHT', 1:nx, ny-10:ny, []);
ix4 = searchForBoundaryFaces(Gt, 'FRONT', 1:nx/2-8, ny/2:ny, []);
bcIxVE = [ix1; ix2; ix3; ix4];
clear ix1 ix2 ix3 ix4 nx ny nz

%% Fluid
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
inj_rate  = 5 * mega * 1e3 / year / rhoc;
max_bhp   = 5000*psia; %[]; %10000 * psia;

num_wells = randi([1,3]);
increment = 8072;

actnum      = reshape(grdecl.ACTNUM, G.cartDims);
actnum_l1   = actnum(:,:,1);
well_loc_l1 = randsample(find(actnum_l1(:)), num_wells);

well_locs = zeros(5, num_wells);
for i=1:num_wells
   well_locs(:,i) = (well_loc_l1(i) + (0:4)*increment)';
end

W = [];
for i=1:num_wells
   W = addWell(W, G, rock, well_locs(:,i), ...
               'name'        , ['Injector', int2str(i)] , ...
               'sign'        , 1                        , ...
               'InnerProduct', 'ip_tpf'                 , ...
               'type'        , 'rate'                   , ...
               'val'         , inj_rate / num_wells     , ...
               'lims'        , max_bhp                  , ...
               'comp_i'      , [0 1]);
end
W2D = convertwellsVE(W, G, Gt, rock2D);

%% Vertical Equilibrium simulation
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

VE_model                = CO2VEBlackOilTypeModel(Gt, rock2D, VE_fluid);
[VE_wellSol, VE_states] = simulateScheduleAD(VE_initState, VE_model, VE_schedule);
VE_states               = [{VE_initState} VE_states(:)'];

%% Plot
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