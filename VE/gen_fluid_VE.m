function [fluid_VE] = gen_fluid_VE(rock, Gt, transMult)

    rhow    = 1000;
    co2     = CO2props();                      
    p_ref   = 30 *mega*Pascal;                
    t_ref   = 94+273.15;
    
    co2_rho = co2.rho(p_ref, t_ref);
    co2_c   = co2.rhoDP(p_ref, t_ref) / co2_rho;
    wat_c   = 0;                            
    c_rock  = 4.35e-5 / barsa;                
    srw     = 0.27;                           
    src     = 0.20;                             
    pe      = 5 * kilo * Pascal;           
    muw     = 8e-4 * Pascal * second;           
    muco2   = co2.mu(p_ref, t_ref) * Pascal * second;
    
    invPc3D  = @(pc) (1-srw) .* (pe./max(pc, pe)).^2 + srw;
    kr3D     = @(s) max((s-src)./(1-src), 0).^2;
    fluid_VE = makeVEFluid(Gt, rock, 'P-scaled table'             , ...
                   'co2_mu_ref'  , muco2, ...
                   'wat_mu_ref'  , muw, ...
                   'co2_rho_ref' , co2_rho                , ...
                   'wat_rho_ref' , rhow                   , ...
                   'co2_rho_pvt' , [co2_c, p_ref]         , ...
                   'wat_rho_pvt' , [wat_c, p_ref]         , ...
                   'residual'    , [srw, src]             , ...
                   'pvMult_p_ref', p_ref                  , ...
                   'pvMult_fac'  , c_rock                 , ...
                   'invPc3D'     , invPc3D                , ...
                   'kr3D'        , kr3D                   , ...
                   'transMult'   , transMult);
end

