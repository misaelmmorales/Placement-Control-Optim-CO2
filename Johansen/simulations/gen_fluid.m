function [fluid] = gen_fluid()
    %% Fluid model
    rhow    = 1000;
    co2     = CO2props();
    p_ref   = 30 * mega * Pascal;
    t_ref   = 94 + 273.15;
    rhoc    = co2.rho(p_ref, t_ref);
    cf_co2  = co2.rhoDP(p_ref, t_ref) / rhoc;
    cf_wat  = 0;
    cf_rock = 4.35e-5 / barsa;
    muw     = 8e-4 * Pascal * second;
    muco2   = co2.mu(p_ref, t_ref) * Pascal * second;
    fluid = initSimpleADIFluid('phases', 'WG'           , ...
                               'mu'  , [muw, muco2]     , ...
                               'rho' , [rhow, rhoc]     , ...
                               'pRef', p_ref            , ...
                               'c'   , [cf_wat, cf_co2] , ...
                               'cR'  , cf_rock          , ...
                               'n'   , [2 2]);
    srw = 0.27;
    src = 0.20;
    fluid.krW = @(s) fluid.krW(max((s-srw)./(1-srw), 0));
    fluid.krG = @(s) fluid.krG(max((s-src)./(1-src), 0));
    pe = 5 * kilo * Pascal;
    pcWG = @(sw) pe * sw.^(-1/2);
    fluid.pcWG = @(sg) pcWG(max((1-sg-srw)./(1-srw), 1e-5)); 

end

