clear; close all; clc
mrstModule add ad-core ad-props co2lab coarsegrid
set(0,'DefaultFigureWindowStyle','docked')

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
pe      = 5 *kilo*Pascal;                       % capillary entry pressure
muw     = 8e-4 *Pascal*second;                  % brine viscosity
muco2   = co2.mu(p_ref, t_ref) *Pascal*second;  % co2 viscosity

%% Grid, petrophysical data, well, and initial state
% Grid and open boundary faces
[G, rock, bcIx, ~, ~, bcIxVE] = makeJohansenVEgrid();
max_perm = max(rock.perm);

W = [];
Wm = [];
mon_num = randi([1,5]);

% Setup the well
while isempty(W)
    inj_idx = randi([25,75], [1,2]);
    wc_global = false(G.cartDims); 
    wc_global(inj_idx(1), inj_idx(2), 6:10) = true;
    wc = find(wc_global(G.cells.indexMap));

    inj_rate  = 3.5 * mega * 1e3 / year / co2_rho;
    W = addWell([], G, rock, wc, 'name',   'Injector', ...
                                 'type',   'rate', ...   %constant rate inj
                                 'val',    inj_rate, ... %injection rate vol
                                 'comp_i', [0 1]);       %inject CO2, not water
    while size(Wm,1)<mon_num
        mon_idx = randi(100, [mon_num, 2]);
        for i=1:mon_num
            mc_global = false(G.cartDims);
            mc_global(mon_idx(i,1), mon_idx(i,2), :) = true;
            mc = find(mc_global(G.cells.indexMap));
        
            max_perm = 1.5*max(rock.perm);
            rand_perm = max_perm + (3*max_perm-max_perm)*rand;
            rock.perm(mc,:) = repmat(rand_perm, size(mc,1), 1);

            Wm = addWell(Wm, G, rock, mc, 'name',['Monitor ', int2str(i)]);
        end
    end
end

% Top surface grid, petrophysical data, well, and initial state
[Gt, G, transMult] = topSurfaceGrid(G);
rock2D             = averageRock(rock, Gt);
W2D                = convertwellsVE(W, G, Gt, rock2D);
initState.pressure = rhow * g(3) * Gt.cells.z;
initState.s        = repmat([1, 0], Gt.cells.num, 1);
initState.sGmax    = initState.s(:,2);

figure;
plotCellData(G, rock.poro); 
[~,ht]=plotWell(G,W,'color','k'); 
set(ht,'FontSize',10, 'BackgroundColor',[.8 .8 .8]); view(-70,30); 
colormap jet; h=colorbar; h.Label.String='Porosity [v/v]'; 
xlabel('X'); ylabel('Y'); zlabel('Z'); title('Porosity')
[~, ht2] = plotWell(G, Wm, 'color', 'r');
set(ht2,'FontSize',10, 'BackgroundColor',[.8 .8 .8])

figure;
plotCellData(G, log10(convertTo(rock.perm(:,1), milli*darcy))); 
[~,ht]=plotWell(G,W,'color','k'); 
set(ht,'FontSize',10, 'BackgroundColor',[.8 .8 .8]); view(-70,30); 
colormap jet; h=colorbar; h.Label.String='log(k_x) [mD]'; 
xlabel('X'); ylabel('Y'); zlabel('Z'); title('Permeability')
[~, ht2] = plotWell(G, Wm, 'color', 'r');
set(ht2,'FontSize',10, 'BackgroundColor',[.8 .8 .8])

%% Special Visualization
% To avoid plotting artifacts when visualizing the volumetric and the
% top-surface grids, we shift the volumetric grid 100 meters down,
%{
GG = G; 
GG.nodes.coords(:,3) = GG.nodes.coords(:,3) + 100;
screensize = get(0,'screensize'); 
figure('position',[screensize(3:4)-[845 565] 840 480]);
plotGrid(GG, 'facecolor', [1 1 .7]); 
plotGrid(Gt, 'facecolor', [.4 .5 1]);
[~,ht]=plotWell(G,W, 'color', 'k'); 
set(ht,'FontSize',10, 'BackgroundColor',[.8 .8 .8]);
view(-70,30); colorbar; clear GG;
%}

%% Fluid model
invPc3D = @(pc) (1-srw) .* (pe./max(pc, pe)).^2 + srw;
kr3D    = @(s) max((s-src)./(1-src), 0).^2; % uses CO2 saturation
fluid   = makeVEFluid(Gt, rock, 'P-scaled table', ... %lin-scale Cap Fringe
               'co2_mu_ref'  , muco2, ...%6e-5 * Pascal * second , ...
               'wat_mu_ref'  , muw, ...  %8e-4 * Pascal * second , ...
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
schedule.step.val     = [repmat(year,50, 1); repmat(10*year, 95, 1)];
schedule.step.control = [ones(50, 1);        ones(95, 1) * 2];

%% Create and simulate model
model = CO2VEBlackOilTypeModel(Gt, rock2D, fluid);
[wellSol, states] = simulateScheduleAD(initState, model, schedule);
states = [{initState} states(:)'];

%% Animate the plume migration over the whole simulation period
figure
oG = generateCoarseGrid(Gt.parent, ones(Gt.parent.cells.num,1));
plotFaces(oG, 1:oG.faces.num,'FaceColor','none');
plotWell(Gt.parent, W,'FontSize',10);
plotWell(Gt.parent, Wm, 'color' ,'r');
view(-70, 30); colorbar, clim([0,1]); colormap jet;
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
                                    'p', states{45}.pressure);

    sat = height2Sat(struct('h', h, 'h_max', h_max), Gt, fluid);
    title(sprintf('Time: %4d yrs (%s)', time(i),ptxt{period(i)}));
    ix = sat>0; if ~any(ix), continue; end
    hs = plotCellData(Gt.parent, sat, ix); drawnow
end


%%
hs     = [];
time   = cumsum([0; schedule.step.val])/year;
period = [1; schedule.step.control];
ptxt   = {'injection','migration'};
figure; view(-70,30); colormap jet
plotWell(G, Wm); plotWell(G, W)
for i=1:numel(states)
    delete(hs)
    hs = plotCellData(Gt, states{1,i}.s(:,2)); drawnow
    title(sprintf('Time: %4d yrs (%s)', time(i),ptxt{period(i)}));
end

%% Trapping inventory
%{
ta = trapAnalysis(Gt, false);
reports = makeReports(Gt, states, model.rock, model.fluid, ...
                      schedule, [srw, src], ta, []);

h1 = figure; plot(1); ax = get(h1, 'currentaxes');
plotTrappingDistribution(ax, reports, 'legend_location', 'northwest');
%}
%% END