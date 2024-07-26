function [rock, rock2D] = gen_rock_VE(realization, G, Gt)
    r = load(sprintf('data_100_100_11/rock/rock_%d.mat', realization));
    p = r.poro(:);
    K = 10.^r.perm(:);  

    rock.perm = bsxfun(@times, [1 1 0.1], K(G.cells.indexMap)).*milli*darcy;
    rock.poro = p(G.cells.indexMap);
    clear p K;

    rock2D = averageRock(rock, Gt);
end

