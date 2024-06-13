%% Main variables
proj_dir = pwd;
mdir = 'mrst-2023a';
chdir(mdir); startup; chdir(proj_dir);
clear; clc; close all

mrstModule add ad-core ad-props co2lab coarsegrid mrst-gui linearsolvers

%% VE Runner
n_realizations = 318*4;

parfor i=1:n_realizations
    [VE_states, ta_reports] = simulationVE(i-1)

    sname = sprintf('data_100_100_11/VE_states/states_%d', i-1);
    parsave(sname, VE_states)
    
    rname = sprintf('data_100_100_11/VE_reports/reports_%d', i-1);
    parsave(rname, ta_reports)

    fprintf('Simulation %i done\n', i)

end

%% END