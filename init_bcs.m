nx = G.cartDims(1); ny=G.cartDims(2); nz=G.cartDims(3);
ix1 = searchForBoundaryFaces(G, 'BACK', 1:nx-6, 1:4, 1:nz);
ix2 = searchForBoundaryFaces(G, 'LEFT', 1:20,   1:ny, 1:nz);
ix3 = searchForBoundaryFaces(G, 'RIGHT', 1:nx, ny-10:ny, 1:nz);
ix4 = searchForBoundaryFaces(G, 'FRONT', 1:nx/2-8, ny/2:ny, 1:nz);
bcIx = [ix1; ix2; ix3; ix4];

nx = Gt.cartDims(1); ny=Gt.cartDims(2);
ix1 = searchForBoundaryFaces(Gt, 'BACK',  1:nx-6, 1:4, []);
ix2 = searchForBoundaryFaces(Gt, 'LEFT',  1:20, 1:ny,  []);
ix3 = searchForBoundaryFaces(Gt, 'RIGHT', 1:nx, ny-10:ny, []);
ix4 = searchForBoundaryFaces(Gt, 'FRONT', 1:nx/2-8, ny/2:ny, []);
bcIxVE = [ix1; ix2; ix3; ix4];
clear ix1 ix2 ix3 ix4 nx ny nz