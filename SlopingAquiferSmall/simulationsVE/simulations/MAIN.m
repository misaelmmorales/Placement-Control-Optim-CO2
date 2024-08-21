%% Main variables
set(0,'DefaultFigureWindowStyle','docked')
% Import MRST module
mrstModule add co2lab mimetic matlab_bgl
mrstModule add ad-core ad-props ad-blackoil mrst-gui
clear;clc

% Define global variables
dims = 64;
gravity on;  g = gravity;

%% Make Grid
[G, Gt, ~, ~, bcIx, bcIxVE] = makeModel(32,1);
%save('grids/G.mat', 'G')
%save('grids/Gt.mat', 'Gt')

figure(1); clf; plotCellData(G, G.cells.centroids(:,3)); view(3); colormap jet; colorbar
figure(2); clf; plotCellData(Gt, Gt.cells.z); view(3); colormap jet; colorbar

%% Make Fluid and BCs
fluidVE = initVEFluidHForm(Gt, 'mu' , [0.056641 0.30860] .* centi*poise, ...
                               'rho', [686.54 975.86] .* kilogram/meter^3, ...
                               'sr', 0.20, ... %residual co2 saturation
                               'sw', 0.10, ... %residual water saturation
                               'kwm', [0.2142 0.85]);

bcVE     = addBC([], bcIxVE, 'pressure', Gt.faces.z(bcIxVE)*fluidVE.rho(2)*norm(gravity));
bcVE     = rmfield(bcVE,'sat');
bcVE.h   = zeros(size(bcVE.face));

ts     = findTrappingStructure(Gt);
p_init = 3500*psia;

%% Generate Models
perm = load('data/perm_64x64.mat').perm';
poro = load('data/poro_64x64.mat').poro';
facies = load('data/facies_64x64.mat').facies';

timesteps  = [2010*year(), 10*year(), 0.5*year(), 100*year()];
time_arr = [repmat(timesteps(3)/year, timesteps(2)/timesteps(3), 1); 
            repmat(timesteps(4)/year, (timesteps(1)-timesteps(2))/timesteps(4), 1)];
save('data/time_arr.mat', 'time_arr')

total_inj  = (35 / (timesteps(2)/year) ); % 35 MT over 10 yrs = 3.5 MT/yr
min_inj    = 0.25; % in MT CO2
conversion = fluidVE.rho(1) * (year/2) / 1e3 / mega;

%% Run Simulation
tic
parfor i=0:999   
    [rock]               = gen_rock(i, perm, poro, facies);
    [W, WVE, wellIx]     = gen_wells(G, Gt, rock);
    [controls]           = gen_controls(timesteps, total_inj, min_inj, W, fluidVE);
    [schedule]           = gen_schedule(timesteps, W, bcVE, controls);
    [SVE, preComp, sol0] = gen_init(Gt, rock, fluidVE, W, p_init);
    [states]             = gen_simulation(timesteps, sol0, Gt, rock, WVE, ...
                                            controls, fluidVE, bcVE, ...
                                            SVE, preComp, ts)
    parsave(sprintf('states/states_%d', i), states);
    parsave(sprintf('controls/controls_%d', i), controls*conversion);
    parsave(sprintf('well_locs/well_locs_%d', i), wellIx);
    parsave(sprintf('rock/rock_%d', i), rock);
    parsave(sprintf('porevol/pv_%d', i), poreVolume(G,rock));
    fprintf('Simulation %i done\n', i)
end
toc; disp('... All Done!');

%% END