%% VE-incomp: Sloping Aquifer Big
% Set time parameters:
% Inject CO2 for 100 years and study subsequent migration for 1500 years
% until 1600 years after injection started. 
% Steps: inj=10, migra
% tion=30.
% Max injection: 20 MT CO2 per year per well
% The fluid data are chosen so that they are resonable at p = 300 bar
% timesteps = [T, stopInj, dT, dT2]
mrstModule add co2lab-common co2lab-spillpoint co2lab-legacy
mrstModule add mimetic matlab_bgl mrst-gui
clear;clc; %close all
gravity on

[G, Gt, ~, ~, bcIxVE] = makeSlopingAquiferBig(true);

fluidVE = initVEFluidHForm(Gt, ...
                           'mu' , [0.056641 0.30860] .* centi*poise, ...
                           'rho', [686.54   975.86]  .* kilogram/meter^3, ...
                           'sr' , 0.20, ... %residual co2 saturation
                           'sw' , 0.27, ... %residual water saturation
                           'kwm', [0.2142 0.85]);
ts        = findTrappingStructure(Gt);
p_init    = 300*barsa(); % ~ 4351 psia
bcVE      = addBC([], bcIxVE, 'pressure', Gt.faces.z(bcIxVE)*fluidVE.rho(2)*norm(gravity));
bcVE      = rmfield(bcVE,'sat');
bcVE.h    = zeros(size(bcVE.face));
timesteps = [1530*year(), 30*year(), 1*year(), 50*year()];
total_inj = (2000/30); % in MT CO2 ==> 2 GT over 30 years
min_inj   = 3; % in MT CO2

%% Main loop
parpool(8)

parfor (i=1:1272)
    [rock, rock2D]       = gen_rock(G, Gt, i-1);
    [W, WVE, wellIx]     = gen_wells(G, Gt, rock2D);
    [controls]           = gen_controls(timesteps, total_inj, min_inj, W, fluidVE);
    [SVE, preComp, sol0] = gen_init(Gt, rock2D, fluidVE, W, p_init);
    [states]             = gen_simulation(timesteps, sol0, Gt, rock2D, ...
                                          WVE, controls, fluidVE, bcVE, ...
                                          SVE, preComp, ts);
    
    parsave(sprintf('states/states_%d', i-1), states);
    parsave(sprintf('controls/controls_%d', i-1), controls);
    parsave(sprintf('well_locs/well_locs_%d', i-1), wellIx);
    parsave(sprintf('rock/VE2d/rock2d_%d', i-1), rock2D);
    fprintf('Simulation %i done\n', i-1)

end

%% END

%{
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