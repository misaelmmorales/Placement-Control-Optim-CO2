function [G, rock, bcIx, Gt, transMult, rock2D, bcIxVE, grdecl] = make_VEJohansen(i)
%Make a VE model based upon a data set of the Johansen formation
%   G      - Data structure for 3D grid
%   Gt     - Data structure for topsurface grid
%   rock   - Data structure for 3D rock parameters
%   rock2D - Data structure for rock parameters for topsurface grid
%   bcIxVE - Index for pressure boundary conditions in topsurface grid

sector = fullfile(mrstPath('co2lab'), 'data', 'johansen', 'NPD5');
grdecl = readGRDECL([sector '.grdecl']);

% Load permeability and porosity
fname = sprintf('data_100_100_11/rock/rock_%d.mat', i);
r = load(fname);
p = r.poro(:);
K = 10.^r.perm(:);

% Construct grid structure.
G = processGRDECL(grdecl);
G = computeGeometry(G(1));
[Gt, G, transMult] = topSurfaceGrid(G);

% Construct structure with petrophyiscal data.
rock.perm = bsxfun(@times, [1 1 0.1], K(G.cells.indexMap)).*milli*darcy;
rock.poro = p(G.cells.indexMap);
rock2D    = averageRock(rock, Gt);
clear p K;

%% FIND PRESSURE BOUNDARY
% Setting boundary conditions is unfortunately a manual process and may
% require some fiddling with indices, as shown in the code below. Here, we
% identify the part of the outer boundary that is open, i.e., not in
% contact with one of the shales (Dunhil or Amundsen).

% boundary 3D
nx = G.cartDims(1); ny=G.cartDims(2); nz=G.cartDims(3);
ix1 = searchForBoundaryFaces(G, 'BACK', 1:nx-6, 1:4, 1:nz);
ix2 = searchForBoundaryFaces(G, 'LEFT', 1:20,   1:ny, 1:nz);
ix3 = searchForBoundaryFaces(G, 'RIGHT', 1:nx, ny-10:ny, 1:nz);
ix4 = searchForBoundaryFaces(G, 'FRONT', 1:nx/2-8, ny/2:ny, 1:nz);
bcIx = [ix1; ix2; ix3; ix4];

% boundary 2D
nx = Gt.cartDims(1); ny=Gt.cartDims(2);
ix1 = searchForBoundaryFaces(Gt, 'BACK',  1:nx-6, 1:4, []);
ix2 = searchForBoundaryFaces(Gt, 'LEFT',  1:20, 1:ny,  []);
ix3 = searchForBoundaryFaces(Gt, 'RIGHT', 1:nx, ny-10:ny, []);
ix4 = searchForBoundaryFaces(Gt, 'FRONT', 1:nx/2-8, ny/2:ny, []);
bcIxVE = [ix1; ix2; ix3; ix4];
clear ix1 ix2 ix3 ix4 nx ny nz

end