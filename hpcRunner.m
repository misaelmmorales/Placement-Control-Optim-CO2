%% Main variables
proj_dir = pwd;
mdir = 'work/08649/mmm6558/ls6/Placement-Control-Optim-CO2/mrst-2023a';
chdir(mdir); startup; chdir(proj_dir);
clear; clc; close all

mrstModule add ad-core ad-props co2lab coarsegrid mrst-gui linearsolvers ad-blackoil

%% Runner
n_realizations = 1272;

parfor i=1:n_realizations
    [states] = simulation(i-1)

    sname = sprintf('states/states_%d', i-1);
    parsave(sname, states)

    fprintf('Simulation %i done\n', i)

end

%% END