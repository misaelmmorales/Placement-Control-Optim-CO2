%% Main variables
proj_dir = pwd;
mdir = 'C:/Users/Misael Morales/OneDrive - The University of Texas at Austin/Documents/MATLAB/mrst-2022b';
chdir(mdir); startup; chdir(proj_dir);
clear; clc; close all

set(0,'DefaultFigureWindowStyle','docked')
mrstModule add ad-core ad-props co2lab coarsegrid mrst-gui

%% Initialize Grid, Rock, and BCs
[G, rock, bcIx, Gt, transMult, rock2D, bcIxVE, grdecl] = make_VEJohansen(99);

init_fluid

initState.pressure = rhow * g(3) * Gt.cells.z;
initState.s        = repmat([1, 0], Gt.cells.num, 1);
initState.sGmax    = initState.s(:,2);

p_bc     = Gt.faces.z(bcIxVE) * rhow * g(3);
bc2D     = addBC([], bcIxVE, 'pressure', p_bc); 
bc2D.sat = repmat([1 0], numel(bcIxVE), 1);

%% VE Fluid
invPc3D = @(pc) (1-srw) .* (pe./max(pc, pe)).^2 + srw;
kr3D    = @(s) max((s-src)./(1-src), 0).^2;
fluid = makeVEFluid(Gt, rock, 'P-scaled table' , ...
               'co2_mu_ref'  , muco2           , ...
               'wat_mu_ref'  , muw             , ...
               'co2_rho_ref' , co2_rho         , ...
               'wat_rho_ref' , rhow            , ...
               'co2_rho_pvt' , [co2_c, p_ref]  , ...
               'wat_rho_pvt' , [wat_c, p_ref]  , ...
               'residual'    , [srw, src]      , ...
               'pvMult_p_ref', p_ref           , ...
               'pvMult_fac'  , c_rock          , ...
               'invPc3D'     , invPc3D         , ...
               'kr3D'        , kr3D            , ...
               'transMult'   , transMult);

%% Well(s)
inj_rate  = 3 * mega * 1e3 / year / co2_rho;
max_bhp   = 5000 * psia;

num_wells = randi([1,3]);

increment = 8072;
actnum = reshape(grdecl.ACTNUM, [100,100,11]);
actnum_l1 = actnum(:,:,1);
well_loc_l1 = randsample(find(actnum_l1(:)), num_wells);

well_locs = zeros(5,num_wells);
for i=1:num_wells
    well_locs(:,i) = (well_loc_l1(i) + (0:4)*increment)';
end

W   = []; 
W   = make_wells(W, G, rock, well_locs, inj_rate, max_bhp);
W2D = convertwellsVE(W, G, Gt, rock2D);

%% Schedule
% total injection of 30 Mt CO2 over 10 years
% change controls every 6 months (20 control steps)

for i=1:20
    schedule.control(i) = struct('W', W2D, 'bc', bc2D);
end
schedule.control(21) = struct('W', W2D, 'bc', bc2D);

min_rate  = 0.5 * mega * 1e3 / year / co2_rho;
well_rate = 5.0*rand(1,20) * mega * 1e3 / year / co2_rho;
well_rate(well_rate<min_rate) = 0;
well_rate = well_rate * (30/(co2_rho*sum(well_rate*year/2)/mega/1e3));

for i=1:num_wells
    schedule.control(2).W(i).val = 0;
    for k=1:20
        schedule.control(k).W(i).val = well_rate(k);
    end
end

schedule.step.val     = [repmat(year/2,20,1); repmat(10*year,50,1)];
schedule.step.control = [linspace(1,20,20)';  ones(50,1)*21];

%% Run simulation
model             = CO2VEBlackOilTypeModel(Gt, rock2D, fluid);
[wellSol, states] = simulateScheduleAD(initState, model, schedule);
states            = [{initState} states(:)'];

%% Trap analysis
ta      = trapAnalysis(Gt, false);
reports = makeReports(Gt, states, model.rock, model.fluid, schedule, [srw, src], ta, []);

%% Plots
plot_simulation_animation
plot_trapping_inventory

%% END