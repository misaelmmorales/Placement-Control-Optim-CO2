function [controls] = gen_controls(timesteps, max_inj, W, fluidVE)

    % Extract time parameters:
    T          = timesteps(1);
    stopInject = timesteps(2);
    dT         = timesteps(3);
    dT2        = timesteps(4);
    nTinj      = (stopInject/year) / (dT/year);

    num_wells = size(W,1);

    controls = max_inj * rand(num_wells, nTinj);
    controls(controls < 1) = 0;
    controls = controls .* ((max_inj*nTinj) ./ sum(controls, 2));
    controls = controls * 1e3 * mega / fluidVE.rho(1) / year;
    
end

% rate = 30 * 1e3 * mega / fluidVE.rho(1) / year;  %2.8e4*meter^3/day