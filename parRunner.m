%% Main variables
proj_dir = pwd;
mdir = 'C:/Users/Misael Morales/OneDrive - The University of Texas at Austin/Documents/MATLAB/mrst-2022b';
chdir(mdir); startup; chdir(proj_dir);
clear; clc; close all

set(0,'DefaultFigureWindowStyle','docked')
mrstModule add ad-core ad-props ad-blackoil co2lab coarsegrid mrst-gui linearsolvers
gravity on;

%% setup
[G, ~, bcIx, ~, ~, ~] = makeJohansenVEgrid();
g = gravity;
rhow = 1000;
initState.pressure = rhow * g(3) * G.cells.centroids(:,3);
initState.s = repmat([1, 0], G.cells.num, 1);
initState.sGmax = initState.s(:,2);
fluid = gen_fluid();
bc = [];
p_bc = rhow * g(3) * G.faces.centroids(bcIx, 3);
bc = addBC(bc, bcIx, 'pressure', p_bc, 'sat', [1, 0]);

%% RUNNER
n_realizations = 318*4;
parfor i=1:n_realizations
    rock        = gen_rock(i-1, G);
    [W,x,y]     = gen_wells(G, rock);
    [schedule]  = gen_schedule(W, bc, fluid);
    [~, states] = gen_simulation(G, rock, fluid, initState, schedule);
    parsave(sprintf('data_100_100_11/states/states_%d', i-1), states)
    parsave(sprintf('data_100_100_11/well_coords/well_coords_%d', i-1), struct('X',x,'Y',y));
    fprintf('Simulation %i done\n', i)
end

%% END

% figure(1); clf; 
% plotCellData(G, rock.poro); plotWell(G,W); colormap jet; colorbar; view(-63,50)

% figure(2); clf; 
% plotToolbar(G, states); plotWell(G,W); colormap jet; colorbar; view(-63,50)

% grdecl = readGRDECL([fullfile(mrstPath('co2lab'),'data','johansen','NPD5'),'.grdecl']);

%     xl = [8; 10; 49];
%     yl = [3; 92; 14];
%     WL = [];
%     for i=1:3
%         wg = false(G.cartDims);
%         wg(xl(i), yl(i), :) = true;
%         wc = find(wg(G.cells.indexMap));
%         wp = g(3) * rhow * G.cells.centroids(wc,3);
%         WL = verticalWell(WL, G, rock, xl(i), yl(i), []    , ...
%                           'name'  , ['Legacy', int2str(i)] , ...
%                           'sign'  , 0                      , ...
%                           'type'  , 'bhp'                  , ...
%                           'val'   , wp(end)                , ...       
%                           'compi' , [0 1]);
%     end