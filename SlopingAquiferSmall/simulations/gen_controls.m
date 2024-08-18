function [controls] = gen_controls(timesteps, total_inj, min_inj, W, fluid)

    % Extract time parameters:
    T          = timesteps(1);
    stopInject = timesteps(2);
    dT         = timesteps(3);
    dT2        = timesteps(4);
    nTinj      = (stopInject/year) / (dT/year);

    num_wells = size(W,1);

    controls = total_inj * rand(num_wells, nTinj);
    controls(controls < min_inj) = 0;
    controls = controls * ((total_inj*nTinj) ./ sum(controls, 'all'));
    controls = controls * 1e3 * mega / fluid.rhoGS / year;
    
end

% rate = 30 * 1e3 * mega / fluidVE.rho(1) / year;  %2.8e4*meter^3/day