function [W, WVE, wellIx] = gen_wells(G, Gt, rock2D)
% Set well and boundary conditions
% We use one well placed down the flank of the model, perforated in the
% bottom layer. Injection rate is 2.8e4 m^3/day of supercritical CO2.
% Hydrostatic boundary conditions are specified on all outer boundaries.

    num_wells = randi([1,5]);
    wellIx    = randi([4,36], [num_wells, 2]);

    rock = rock2D.parent;

    refDepth = G.cells.centroids(G.cells.num/2, G.griddim);

    W = [];
    for i=1:num_wells
        W = verticalWell(W, G, rock, wellIx(i,1), wellIx(i,2), G.cartDims(3), ...
                         'InnerProduct' , 'ip_simple' , ...
                         'Type'         , 'rate'      , ...
                         'Sign'         , 1           , ...
                         'comp_i'       , [1,0]       , ...
                         'Radius'       , 0.1         , ...
                         'refDepth'     , []           , ...
                         'name'         , ['I', int2str(i)]);
    end

    WVE = convertwellsVE(W, G, Gt, rock2D);

end

