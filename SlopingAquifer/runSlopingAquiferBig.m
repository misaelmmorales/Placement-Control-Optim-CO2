%% VE-incomp: Sloping Aquifer Big
% Set time parameters:
% Inject CO2 for 100 years and study subsequent migration for 1200 years
% until 1300 years after injection started. 
% Steps: inj=10, migration=24.
% Max injection: 20 MT CO2 per year per well
% The fluid data are chosen so that they are resonable at p = 300 bar
% timesteps = [T, stopInj, dT, dT2]

mrstModule add co2lab-common co2lab-spillpoint co2lab-legacy
mrstModule add mimetic matlab_bgl mrst-gui
clear;clc;close all
gravity on

[G, Gt, ~, ~, bcIxVE] = makeSlopingAquiferBig(true);

fluidVE = initVEFluidHForm(Gt, ...
                           'mu' , [0.056641 0.30860] .* centi*poise, ...
                           'rho', [686.54 975.86] .* kilogram/meter^3, ...
                           'sr' , 0.2, ... %residual co2 saturation
                           'sw' , 0.1, ... %residual water saturation
                           'kwm', [0.2142 0.85]);
ts        = findTrappingStructure(Gt);
p_init    = 300*barsa(); % ~ 4351 psia
bcVE      = addBC([], bcIxVE, 'pressure', Gt.faces.z(bcIxVE)*fluidVE.rho(2)*norm(gravity));
bcVE      = rmfield(bcVE,'sat');
bcVE.h    = zeros(size(bcVE.face));
timesteps = [1300*year(), 100*year(), 10*year(), 50*year()];
max_inj   = 20; % in MT CO2

parfor i=1:2
    [rock, rock2D] = gen_rock(G, Gt, i);
    [W, WVE, wellIx] = gen_wells(G, Gt, rock2D);
    [controls] = gen_controls(timesteps, max_inj, W, fluidVE);
    [SVE, preComp, sol0] = gen_init(Gt, rock2D, fluidVE, W, p_init);
    [states] = gen_simulation(timesteps, sol0, Gt, rock2D, WVE, controls, fluidVE, bcVE, SVE, preComp, ts);
    
    parsave(sprintf('states/states_%d', i-1), states);
    parsave(sprintf('controls/controls_%d', i-1), controls);
    parsave(sprintf('well_locs/well_locs_%d', i-1), wellIx);
    fprintf('Simulation %i done\n', i)

end

%% END

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