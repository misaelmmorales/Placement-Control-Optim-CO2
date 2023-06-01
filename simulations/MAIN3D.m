%% Main variables
clear; close all; clc
%set(0,'DefaultFigureWindowStyle','docked')

%{
current_dir = pwd;
mrst_dir = 'mrst-2023a';
cd(mrst_dir)
startup
cd(current_dir)
%}

%% Import MRST module
mrstModule add SPE10 coarsegrid upscaling co2lab mrst-gui
mrstModule add ad-core ad-props ad-blackoil

%% Make Grid
dims = 32; nz=5;
dx=1000*meter; dy=1000*meter; dz=500*meter;

G = cartGrid([dims, dims, nz], [dx, dy, dz]);
G = computeGeometry(G);

G10 = computeGeometry(cartGrid([60,220,5], [dx dy dz]));

%% Make rock
[poro1, perm1] = make_rock_layers(1,2);
poro2 = gaussianField(G10.cartDims, [0.05,0.08], [11,5,1], 2.5);
perm2 = reshape(logNormLayers(G10.cartDims, [0.05, 0.05])*milli*darcy, G10.cartDims);
[poro3, perm3] = make_rock_layers(30,31);

poro10 = cat(3, poro1, poro2(:,:,1), poro3);
permx10 = cat(3, perm1, perm2(:,:,1), perm3);

[poro, permx] = deal(zeros(dims,dims,nz));
for i=1:nz
    poro(:,:,i) = imresize(poro10(:,:,i), [dims,dims], 'nearest');
    permx(:,:,i) = imresize(permx10(:,:,i), [dims,dims], 'nearest'); %antialiasing=false
end
poro = reshape(poro, [], 1)+0.05;
permx = reshape(permx, [], 1);
[perm(:,1), perm(:,2)] = deal(permx);
perm(:,3) = 0.1*permx;
rock.poro = poro;
rock.perm = perm;

%permeability = convertTo(perm, milli*darcy);
clear i poro1 poro2 poro3 perm1 perm2 perm3
clear permeability permx perm poro

%% Make Initial State
gravity on;  g = gravity;
rhow = 1000; % density of brine corresponding to 94 degrees C and 300 bar
%initState.pressure = G.cells.centroids(:,3) * 400*psia; %G.cells.centroids=3
P0 = 4000*psia; 
initState.pressure = repmat(P0, dims*dims*nz, 1);
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
inj_time = 5*year;
inj_timesteps = rampupTimesteps(inj_time, year/12, 10);

mon_time = 5*year;
mon_timesteps = rampupTimesteps(mon_time, year, 0);

total_time = inj_time + mon_time;
cum_time = convertTo(cumsum([inj_timesteps; mon_timesteps]), year);

irate = 0.25*sum(poreVolume(G, rock))/(mon_time);

%% Run Simulations
N_realization = 50;

modrock = cell(N_realization, 1);
results  = cell(N_realization, 1);
inj_locations = cell(N_realization,1);
mon_locations = cell(N_realization,1);

parfor i=1:N_realization
    loc_inj = randi([4,28], [1,2]);
    num_moni = randi(7);
    loc_moni = randi([2,30], [num_moni,2]);

    newrock = gen_monitorwells(G, rock, num_moni, loc_moni);
    W = gen_wells_3d(G, newrock, loc_inj, nz, irate);
    [schedule, dT1, dT2] = gen_schedule_3d(W, bc, inj_timesteps, mon_timesteps);
    [model, wellSol, states] = gen_simulation(G, newrock, fluid, initState, schedule);

    inj_locations{i} = loc_inj;
    mon_locations{i} = loc_moni;

    results{i} = states;
    modrock{i} = newrock;
end
% max(max(reshape(results{20,1}{end,1}.s(:,2), [32,32,5])))
% plot(reshape(max(max(reshape(results{44,1}{end,1}.s(:,2), [32,32,5]))), [], 1))


%% Collect Results
%{
inj_locations = cell2mat(inj_locations);

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
%}

%% Plots

%{
figure
plotCellData(G10, reshape(poro10,[],1)); colormap jet; view(-70,80); colorbar
figure
plotCellData(G10, log10(convertTo(reshape(permx10,[],1), milli*darcy))); colormap jet; view(-70,80); colorbar
figure
plotCellData(G, rock.poro); colormap jet; colorbar; view(-70,80)
figure
plotCellData(G, log10(convertTo(rock.perm(:,1),milli*darcy))); colormap jet; colorbar; view(-70,80)

figure
sat_end = results{1,1}{end,1}.s(:,2);  % co2 saturation at end state
plume_cells = sat_end > 0.05;
clf; plotGrid(G, 'facecolor', 'none');
plotGrid(G, plume_cells, 'facecolor', 'red')
view(-70, 80);

figure
plotToolbar(G, results{1}); view(-70,80); colormap jet; colorbar
%}

%% END