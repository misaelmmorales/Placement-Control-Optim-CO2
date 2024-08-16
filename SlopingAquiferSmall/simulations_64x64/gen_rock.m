function [rock, rock2D] = gen_rock(G, Gt, realization)

    r = load(sprintf('E:/Placement-Control-Optim-CO2/SlopingAquiferSmall/rock/mat/rock_%d.mat', realization));
    pp = r.poro(:);
    %k = 10 .^ r.perm(:) * milli * darcy;
    k = convertFrom(r.perm(:), milli*darcy);
    kk(:,1) = k;
    kk(:,2) = k;
    kk(:,3) = 0.1*k;
    rock = makeRock(G, kk, pp);
    rock2D = averageRock(rock, Gt);
    
    clear r pp k kk

end

