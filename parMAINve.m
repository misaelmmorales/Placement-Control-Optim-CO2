%% Main variables
proj_dir = pwd;
mdir = 'C:/Users/Misael Morales/OneDrive - The University of Texas at Austin/Documents/MATLAB/mrst-2022b';
chdir(mdir); startup; chdir(proj_dir);
clear; clc; close all

set(0,'DefaultFigureWindowStyle','docked')
mrstModule add ad-core ad-props co2lab coarsegrid mrst-gui

%% Fluid
gravity reset on;
fluid.g       = gravity;
fluid.co2     = CO2props();                             % CO2 property functions
fluid.p_ref   = 30 *mega*Pascal;                        % reference pressure
fluid.t_ref   = 94+273.15;                              % reference temperature
fluid.rhow    = 1000;                                   % water density (kg/m^3)
fluid.rhoc    = fluid.co2.rho(fluid.p_ref, fluid.t_ref); % CO2 density
fluid.c_co2   = fluid.co2.rhoDP(fluid.p_ref, fluid.t_ref) / fluid.rhoc; % CO2 compressibility
fluid.c_water = 0;                                      % water compressibility
fluid.c_rock  = 4.35e-5 / barsa;                        % rock compressibility
fluid.srw     = 0.27;                                   % residual water
fluid.src     = 0.20;                                   % residual CO2
fluid.pe      = 5 * kilo * Pascal;                      % capillary entry pressure
fluid.muw     = 8e-4 * Pascal * second;                 % brine viscosity
fluid.muco2   = fluid.co2.mu(fluid.p_ref, fluid.t_ref) * Pascal * second; % co2 viscosity

grdecl = readGRDECL([fullfile(mrstPath('co2lab'),'data','johansen','NPD5'),'.grdecl']);

parfor i=1:10
    [G, rock, bcIx, Gt, transMult, rock2D, bcIxVE] = make_Johansen(i, grdecl);
    [W, W2D]                        = make_wells(G, Gt, grdecl, rock, rock2D, fluid, []);
    [VE_fluid]                      = make_VEfluid(fluid, Gt, rock2D, transMult);
    [VE_initState, VE_schedule]     = make_controls(fluid, Gt, bcIxVE, W2D);
    [VE_model,VE_wellSol,VE_states] = make_simulation(Gt, rock2D, VE_fluid, VE_initState, VE_schedule);
end

