function [states] = gen_simulation(tsteps, sol, Gt, rock, WVE, controls, fluidVE, bcVE, SVE, preComp, ts)

% Main loop
% Run the simulation using a sequential splitting with pressure and
% transport computed in separate steps. The transport solver is formulated
% with the height of the CO2 plume as the primary unknown and the relative
% height (or saturation) must therefore be reconstructed.

    % Initialize parameters
    t         = 0;
    totVol    = 0.0;
    count     = 1;
    states    = struct('pressure', [], 's',   [], 's3d',     [], ...
                       't',        [], 'dT',  [], 'wellSol', [], ...
                       'totVol',   [], 'vol', [], ...
                       'freeVol',  [], 'trappedVol', [], 'leakedVol', []);

    % Extract time parameters:
    T          = tsteps(1);
    stopInject = tsteps(2);
    dT         = tsteps(3);
    dT2        = tsteps(4);

    while t<T
        
        if count <= (stopInject/dT)
            for k=1:size(WVE,1)
                WVE(k).val = controls(k, count);
            end
        end

       % Advance solution: compute pressure and then transport
       sol = solveIncompFlowVE(sol, Gt, SVE, rock, fluidVE, ...
                               'bc', bcVE, 'wells', WVE);

       sol = explicitTransportVE(sol, Gt, dT, rock, fluidVE, ...
                                 'bc'      , bcVE    , ...
                                 'wells'   , WVE     , ...
                                 'preComp' , preComp , ...
                                 'intVert' , false);
    
       % Reconstruct 'saturation' defined as s=h/H, where h is the height of
       % the CO2 plume and H is the total height of the formation
       sol.s = height2finescaleSat(sol.h, sol.h_max, Gt, fluidVE.res_water, fluidVE.res_gas);
       assert( max(sol.s(:,1))<1+eps && min(sol.s(:,1))>-eps );
       t = t + dT;
    
       % Compute total injected, trapped and free volumes of CO2
       if ~isempty(WVE)
          totVol = totVol + sum([WVE.val])*dT;
       end
       vol = volumesVE(Gt, sol, rock, fluidVE, ts);
    
       states(count).pressure   = sol.pressure;
       states(count).s          = sol.h ./ Gt.cells.H;
       states(count).s3d        = sol.s;
       states(count).wellSol    = sol.wellSol;
       states(count).totVol     = totVol;
       states(count).vol        = vol;
       states(count).freeVol    = sum(sol.h .* rock.poro .* Gt.cells.volumes)*(1-fluidVE.res_water);
       states(count).trappedVol = sum((sol.h_max - sol.h) .* rock.poro .* Gt.cells.volumes)*fluidVE.res_gas;
       states(count).leakedVol  = max(0, totVol - sum(vol));
       states(count).t          = convertTo(t, year);
       states(count).dT         = convertTo(dT, year);
       
       % Check if we are to stop injecting. If so, increase the time step.
       if t>= stopInject
          WVE    = [];
          dT     = dT2;
       end
    
       count = count+1;
    
    end
    
end