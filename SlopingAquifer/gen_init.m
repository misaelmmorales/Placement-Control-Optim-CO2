function [SVE, preComp, sol] = gen_init(Gt, rock2D, fluidVE, W, p_init)

% Prepare simulations
% Compute inner products and instantiate solution structure

    SVE         = computeMimeticIPVE(Gt, rock2D, 'Innerproduct','ip_simple');
    
    preComp     = initTransportVE(Gt, rock2D);
    
    sol         = initResSolVE(Gt, 0, 0);
    
    sol.wellSol = initWellSol(W, p_init);
    
    sol.s       = height2finescaleSat(sol.h, sol.h_max, Gt, ...
                                        fluidVE.res_water, ...
                                        fluidVE.res_gas);
        
end

