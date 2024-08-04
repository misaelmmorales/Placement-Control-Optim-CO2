function hpc_runSlopingAquiferBig(i)

cd('/work/08649/mmm6558/ls6/mrst-2024a'); 
startup
cd('/work/08649/mmm6558/ls6/Placement-Control-Optim-CO2/SlopingAquifer');

mrstModule add co2lab-common co2lab-spillpoint co2lab-legacy
mrstModule add mimetic matlab_bgl mrst-gui
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

[rock, rock2D]       = gen_rock(G, Gt, i-1);
[W, WVE, wellIx]     = gen_wells(G, Gt, rock2D);
[controls]           = gen_controls(timesteps, total_inj, min_inj, W, fluidVE);
[SVE, preComp, sol0] = gen_init(Gt, rock2D, fluidVE, W, p_init);
[states]             = gen_simulation(timesteps, sol0, Gt, rock2D, ...
                                      WVE, controls, fluidVE, bcVE, ...
                                      SVE, preComp, ts);

%% Save outputs
parsave(sprintf('states/states_%d', i-1), states);
parsave(sprintf('controls/controls_%d', i-1), controls);
parsave(sprintf('well_locs/well_locs_%d', i-1), wellIx);

clc
fprintf('Simulation %i done\n', i-1)

%% END

end