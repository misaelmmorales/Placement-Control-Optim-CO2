%% Main variables
proj_dir = pwd;
mdir = 'C:/Users/Misael Morales/OneDrive - The University of Texas at Austin/Documents/MATLAB/mrst-2022b';
chdir(mdir); startup; chdir(proj_dir);
clear; clc; close all

set(0,'DefaultFigureWindowStyle','docked')
mrstModule add ad-core ad-props co2lab coarsegrid mrst-gui

%% Grid, Rock, BCs

parfor i=1:10
    [VE_states, reports] = VEsimulation(i-1)
    parsave(fprintf('data_100_100_11/states/states_%d', i-1), 'VE_states')
    parsave(fprintf('data_100_100_11/reports/reports_%d', i-1), 'reports')
end

%% END