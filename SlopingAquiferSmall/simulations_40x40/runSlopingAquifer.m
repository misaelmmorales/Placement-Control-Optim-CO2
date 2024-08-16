%% VE-incomp: Sloping Aquifer Small
% Set time parameters:
% The fluid data are chosen so that they are resonable at p = 300 bar
% timesteps = [T, stopInj, dT, dT2]
mrstModule add co2lab-common co2lab-spillpoint co2lab-legacy
mrstModule add mimetic matlab_bgl mrst-gui
clear;clc; %close all
gravity on

[G, Gt, ~, ~, bcIxVE] = makeModel();
figure(1); clf; plotCellData(G, G.cells.centroids(:,3)); colormap jet; colorbar; view(26,15)
save('G.mat', 'G'); save('Gt.mat', 'Gt');

fluidVE = initVEFluidHForm(Gt, ...
                           'mu' , [0.056641 0.30860] .* centi*poise, ...
                           'rho', [686.54   975.86]  .* kilogram/meter^3, ...
                           'sr' , 0.20, ... %residual co2 saturation
                           'sw' , 0.27, ... %residual water saturation
                           'kwm', [0.2142 0.85]);
ts        = findTrappingStructure(Gt);
p_init    = Gt.cells.z * norm(gravity); % 300*barsa(); % ~ 4351 psia

% E, W, N, S
bcVE      = addBC([],   bcIxVE(1:40), 'flux', 0);
bcVE      = addBC(bcVE, bcIxVE(41:80), 'pressure', Gt.faces.z(bcIxVE(41:80))*fluidVE.rho(2)*norm(gravity));
bcVE      = addBC(bcVE, bcIxVE(81:120), 'pressure', Gt.faces.z(bcIxVE(81:120))*fluidVE.rho(2)*norm(gravity));
bcVE      = addBC(bcVE, bcIxVE(121:end), 'flux', 0);

bcVE      = rmfield(bcVE,'sat');
bcVE.h    = zeros(size(bcVE.face));
timesteps = [2010*year(), 10*year(), 0.5*year(), 100*year()];
% in MT CO2 ==> 50 MT over 10 years = 5 MT per year
total_inj = (50 / (timesteps(2)/year) );
min_inj   = 0.2; % in MT CO2
conversion = fluidVE.rho(1) * (year/2) / 1e3 / mega;

%% Main loop
parfor (i=0:1271)
    [rock, rock2D]       = gen_rock(G, Gt, i);
    [W, WVE, wellIx]     = gen_wells(G, Gt, rock2D);
    [controls]           = gen_controls(timesteps, total_inj, min_inj, W, fluidVE);
    [SVE, preComp, sol0] = gen_init(Gt, rock2D, fluidVE, W, p_init);
    [states]             = gen_simulation(timesteps, sol0, Gt, rock2D, ...
                                          WVE, controls, fluidVE, bcVE, ...
                                          SVE, preComp, ts);

    parsave(sprintf('states/states_%d', i), states);
    parsave(sprintf('controls/controls_%d', i), controls*conversion);
    parsave(sprintf('well_locs/well_locs_%d', i), wellIx);
    parsave(sprintf('rock/VE2d/rock2d_%d', i), rock2D);
    parsave(sprintf('rock/poreVolume/pv_%d', i), poreVolume(G,rock));
    parsave(sprintf('rock/poreVolume2d/pv2d_%d', i), poreVolume(Gt, rock2D));
    fprintf('Simulation %i done\n', i)
end
disp('... All Done!');

%% END

%{
    figure(2); clf; plotCellData(Gt, p_init); colormap jet; colorbar; view(25,45)
    figure(3); clf; plotCellData(G, rock.poro); colormap jet; colorbar; view(25,45); plotWell(G,W)
    figure(4); clf; plot(controls' * conversion);
    figure(4); hold on; plot(sum(controls)' * conversion);
    figure(5); clf; plotToolbar(Gt, states); colormap jet; colorbar; view(25,45)

progressQueue = parallel.pool.DataQueue;
hWaitbar = waitbar(0, 'Processing...');
afterEach(progressQueue, @(~) updateWaitbar(hWaitbar, numIterations));
progressCount = 0;

send(progressQueue, i); %inside parfor
close(hWaitbar); %after parfor

function updateWaitbar(hWaitbar, numIterations)
    persistent count;
    if isempty(count)
        count = 0;
    end
    count = count + 1;
    waitbar(count / numIterations, hWaitbar);
end
%}

%{
% Figures
figure(1); clf
alpha = 0.25;
subplot(231)
plotCellData(G, rock.poro, 'edgealpha',alpha)
title('Porosity [v/v]'); colormap jet
subplot(232)
plotCellData(G, rock.poro, 'edgealpha',alpha)
title('Porosity [v/v]'); colormap jet; colorbar('horizontal'); view(-30,45)
subplot(233)
plotCellData(Gt, rock2D.poro, 'edgealpha',alpha)
title('Porosity [v/v]'); colormap jet; colorbar('horizontal'); view(-30,45)

subplot(234)
plotCellData(G, log10(convertTo(rock.perm(:,1), milli*darcy)), 'edgealpha',alpha)
title('LogPerm [mD]'); colormap jet
subplot(235)
plotCellData(G, log10(convertTo(rock.perm(:,1), milli*darcy)), 'edgealpha',alpha)
title('LogPerm [mD]'); colormap jet; colorbar('horizontal'); view(-30,45)
subplot(236)
plotCellData(Gt, log10(convertTo(rock2D.perm(:,1), milli*darcy)), 'edgealpha',alpha)
title('LogPerm [mD]'); colormap jet; colorbar('horizontal'); view(-30,45)

figure(2); clf; 
plotToolbar(Gt, states, 'edgecolor','k','edgealpha',0.25); 
view(-30,80); colorbar;
%}