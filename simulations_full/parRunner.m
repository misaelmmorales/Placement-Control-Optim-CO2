%% Main variables
proj_dir = pwd;
mdir = 'C:/Users/Misael Morales/OneDrive - The University of Texas at Austin/Documents/MATLAB/mrst-2022b';
chdir(mdir); startup; chdir(proj_dir);
clear; clc; close all

set(0,'DefaultFigureWindowStyle','docked')
mrstModule add ad-core ad-props co2lab coarsegrid mrst-gui linearsolvers ad-blackoil

%% Runner
n_realizations = 10;

parfor i=1:n_realizations
    [states] = simulation(i-1)

    sname = sprintf('data_100_100_11/states/states_%d', i-1);
    parsave(sname, states)

    fprintf('Simulation %i done\n', i)

end

%% END