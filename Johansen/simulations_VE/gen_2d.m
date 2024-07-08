function [Gt, transMult, rock2D, W2D, initState] = gen_2d(G, rock, W, props)

    % Top surface grid, petrophysical data, well, and initial state
    [Gt, G, transMult] = topSurfaceGrid(G);
    rock2D             = averageRock(rock, Gt);
    W2D                = convertwellsVE(W, G, Gt, rock2D);

    initState.pressure = props.rhow * props.g(3) * Gt.cells.z;
    initState.s        = repmat([1, 0], Gt.cells.num, 1);
    initState.sGmax    = initState.s(:,2);

end