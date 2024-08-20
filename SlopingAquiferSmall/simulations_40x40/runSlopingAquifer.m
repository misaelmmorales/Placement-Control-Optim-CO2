%% VE-incomp: Sloping Aquifer Small
mrstModule add mrst-gui
mrstModule add mimetic matlab_bgl
mrstModule add ad-core ad-props ad-blackoil
mrstModule add co2lab
%set(0,'DefaultFigureWindowStyle','docked')
clear;clc; %close all
gravity on; g = gravity;

[G, Gt, ~, ~, bcIx, bcIxVE] = makeModel(20,5);
[CG, ~, ~, ~, bcIx2, ~]     = makeModel(20,1);
save('grids/G.mat', 'G')
save('grids/CG.mat','CG')
save('grids/Gt.mat', 'Gt')

%figure(1); clf; plotCellData(G, G.cells.centroids(:,3)); view(3); colormap jet; colorbar

%% Fluid properties
co2     = CO2props();                            % load sampled tables of co2 fluid props
p_ref   = 30 * mega * Pascal;                    % choose reference pressure
t_ref   = 94 + 273.15;                           % choose reference temp [Kelvin]
rhoc    = co2.rho(p_ref,t_ref);                  % co2 density at ref. press/temp
rhow    = 1000;                                  % brine density at 94C, 300bar
cf_co2  = co2.rhoDP(p_ref,t_ref) / rhoc;         % co2 compressibility
cf_wat  = 0;                                     % brine compressibility
cf_rock = 4.35e-5 / barsa;                       % rock compressibility
muw     = 8e-4 * Pascal * second;                % brine viscosity
muco2   = co2.mu(p_ref,t_ref) * Pascal * second; % co2 viscosity

%% Vertical Equilibrium
fluidVE = initVEFluidHForm(Gt, ...
                           'mu' , [muco2 muw] .* centi*poise      , ...
                           'rho', [rhoc  rhow] .* kilogram/meter^3 , ...
                           'sr' , 0.20, ... %residual co2 saturation
                           'sw' , 0.27, ... %residual water saturation
                           'kwm', [0.2142 0.85]);

ts     = findTrappingStructure(Gt);
p_init = Gt.cells.z * norm(gravity);

% north1VE = bcIxVE(84:2:118);
% north2VE = bcIxVE(141:160);
% northVE  = bcIxVE(119:140);
% eastVE   = bcIxVE(42:2:82);
% westVE   = bcIxVE(43:2:118);
% southVE  = bcIxVE(1:41);
% pVEn1 = Gt.faces.z(north1VE)*fluidVE.rho(2)*norm(g);
% pVEn2 = Gt.faces.z(north2VE)*fluidVE.rho(2)*norm(g);
% pVEe  = Gt.faces.z(eastVE)*fluidVE.rho(2)*norm(g);
% pVEs  = Gt.faces.z(southVE)*fluidVE.rho(2)*norm(g);
% bcVE = [];
% bcVE = addBC(bcVE, westVE, 'flux', 0);
% bcVE = addBC(bcVE, southVE, 'pressure', pVEs);
% bcVE = addBC(bcVE, eastVE, 'flux', 0);
% bcVE = addBC(bcVE, northVE, 'flux', 0);
% bcVE = addBC(bcVE, north1VE, 'pressure', pVEn1);
% bcVE = addBC(bcVE, north2VE, 'pressure', pVEn2);
% bcVE = rmfield(bcVE,'sat');
% bcVE.h = zeros(size(bcVE.face));

bcVE   = addBC([], bcIxVE, 'pressure', Gt.faces.z(bcIxVE)*fluidVE.rho(2)*norm(g));
bcVE   = rmfield(bcVE,'sat');
bcVE.h = zeros(size(bcVE.face));

%% 3D initialization
fluid = initSimpleADIFluid('phases', 'WG'                           , ...
                           'mu'  , [fluidVE.mu(2),  fluidVE.mu(1)]  , ...
                           'rho' , [fluidVE.rho(2), fluidVE.rho(1)] , ...
                           'pRef', p_ref                            , ...
                           'c'   , [cf_wat, cf_co2]                 , ...
                           'cR'  , cf_rock                          , ...
                           'n'   , [2 2]);
pe         = 5 * kilo * Pascal;
pcWG       = @(sw) pe * sw.^(-1/2);
fluid.pcWG = @(sg) pcWG(max((1-sg-fluidVE.res_water)./(1-fluidVE.res_water), 1e-5));
fluid.krW  = @(s) fluid.krW(max((s-fluidVE.res_water)./(1-fluidVE.res_water), 0));
fluid.krG  = @(s) fluid.krG(max((s-fluidVE.res_gas)./(1-fluidVE.res_gas), 0));

initState.pressure = fluidVE.rho(2) * norm(g) * CG.cells.centroids(:,3);
initState.s        = repmat([1,0], CG.cells.num, 1);
initState.sGmax    = initState.s(:,2);

% north0  = bcIx2(120:140);
% north1  = bcIx2(82:2:118);
% north2  = bcIx2(141:160);
% eastAD  = bcIx2(44:2:80);
% westAD  = bcIx2(43:2:120);
% southAD = bcIx2(1:42); 
% pADn1 = fluidVE.rho(2) * g(3) * CG.faces.centroids(north1, 3);
% pADn2 = fluidVE.rho(2) * g(3) * CG.faces.centroids(north2, 3);
% pADe  = fluidVE.rho(2) * g(3) * CG.faces.centroids(eastAD, 3);
% pADs  = fluidVE.rho(2) * g(3) * CG.faces.centroids(southAD, 3);
% bc = [];
% bc = addBC(bc, westAD, 'flux', 0, 'sat', [1,0]);
% bc = addBC(bc, southAD, 'pressure', pADs, 'sat', [1,0]);
% bc = addBC(bc, eastAD, 'flux', 0, 'sat', [1,0]);
% bc = addBC(bc, north0, 'flux', 0, 'sat', [1,0]);
% bc = addBC(bc, north1, 'pressure', pADn1, 'sat', [1,0]);
% bc = addBC(bc, north2, 'pressure', pADn2, 'sat', [1,0]);

pbc = fluidVE.rho(2) * norm(g) * CG.faces.centroids(bcIx2, 3);
bc  = addBC([], bcIx2, 'pressure', pbc, 'sat', [1,0]);

%% Main loop
timesteps  = [1010*year(), 10*year(), 0.5*year(), 50*year()];
time_arr = [repmat(timesteps(3)/year, timesteps(2)/timesteps(3), 1); 
            repmat(timesteps(4)/year, (timesteps(1)-timesteps(2))/timesteps(4), 1)];
save('data/time_arr.mat', 'time_arr')

total_inj  = (10 / (timesteps(2)/year) ); % 10 MT over 10 yrs = 1 MT/yr
min_inj    = 0.2; % in MT CO2
conversion = fluidVE.rho(1) * (year/2) / 1e3 / mega;

parfor (i=0:9)
    % setup
    [rock, rock2d]       = gen_rock(G, Gt, i);
    [W, W2, WVE, wellIx] = gen_wells(G, CG, Gt, rock2d);
    [controls]           = gen_controls(timesteps, total_inj, min_inj, W, fluidVE);
    
    % AD-3D
    [schedule]           = gen_schedule(timesteps, W2, bc, controls);
    [wellSol, states2]   = gen_ADsimulation(CG, rock2d, fluid, initState, schedule);
    
    % VE
    [SVE, preComp, sol0] = gen_init(Gt, rock2d, fluidVE, W, p_init);
    [states]             = gen_simulation(timesteps, sol0, Gt, rock2d, ...
                                          WVE, controls, fluidVE, bcVE, ...
                                          SVE, preComp, ts);
    
    % save outputs
    parsave(sprintf('states/VE2d/states_%d', i), states);
    parsave(sprintf('states/AD2d/states_%d', i), states2);
    parsave(sprintf('controls/controls_%d', i), controls*conversion);
    parsave(sprintf('well_locs/well_locs_%d', i), wellIx);
    parsave(sprintf('rock/VE2d/rock2d_%d', i), rock2d);
    parsave(sprintf('rock/poreVolume/pv_%d', i), poreVolume(G,rock));
    parsave(sprintf('rock/poreVolume2d/pv2d_%d', i), poreVolume(Gt, rock2d));
    fprintf('Simulation %i done\n', i)
    
end
disp('... All Done!');

%% END

%{
bcVE = addBC([],   bcIxVE(1:40), 'flux', 0);
bcVE = addBC(bcVE, bcIxVE(41:80), 'pressure', Gt.faces.z(bcIxVE(41:80))*fluidVE.rho(2)*norm(gravity));
bcVE = addBC(bcVE, bcIxVE(81:120), 'pressure', Gt.faces.z(bcIxVE(81:120))*fluidVE.rho(2)*norm(gravity));
bcVE = addBC(bcVE, bcIxVE(121:end), 'flux', 0);
%}

%{
figure(1); clf; plotCellData(G, rock.poro); plotWell(G,W); view(-25,60); colormap jet; colorbar
figure(2); clf; plotCellData(CG, rock2d.poro); plotWell(CG,W2); view(-25,60); colormap jet; colorbar
figure(3); plotToolbar(CG,states2); plotWell(CG,W2); colormap jet; colorbar; view(-25,60)
figure(4); clf; plotCellData(Gt, rock2d.poro); view(-25,60); colormap jet; colorbar
figure(5); clf; plotToolbar(Gt, states); view(-25,60); colormap jet; colorbar
%}

%{
free    = zeros(40,1);
trapped = zeros(40,1);
leaked  = zeros(40,1);
totvol  = zeros(40,1);
conversion = fluidVE.rho(1) / 1e3 / mega;
tsteps = cumsum([repmat(timesteps(3),20,1); repmat(timesteps(4),20,1)]/year);
for i=1:40
    free(i) = states(i).freeVol * conversion;
    trapped(i) = states(i).trappedVol * conversion;
    leaked(i) = states(i).leakedVol * conversion;
    totvol(i) = states(i).totVol * conversion;
end

figure(6); clf; 
plot(tsteps, totvol, 'color', 'k', 'marker', 'o'); hold on
plot(tsteps, free, 'color','blue', 'marker', 's')
plot(tsteps, trapped, 'color', 'green', 'marker', 'v')
plot(tsteps, leaked, 'color', 'red', 'marker', 'x')
legend('total','free','trapped','leaked', 'Location', 'northwest')
grid; xscale('log'); xlim([0.5,2010])
%}