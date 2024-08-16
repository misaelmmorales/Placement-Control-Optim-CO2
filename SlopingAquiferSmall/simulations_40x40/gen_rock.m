function [rock, rock2D] = gen_rock(G, Gt, realization)

    r = load(sprintf('rock/mat/rock_%d.mat', realization));
    pp = r.poro(:);
    k = convertFrom(10.^r.perm(:), milli*darcy);
    kk(:,1) = k;
    kk(:,2) = k;
    kk(:,3) = 0.1*k;
    rock = makeRock(G, kk, pp);
    rock2D = averageRock(rock, Gt);
    
    clear r pp k kk

end

