function [schedule] = gen_schedule(W2D, bc2D)
    % Setting up two copies of the well and boundary specifications. 
    % Modifying the well in the second copy to have a zero flow rate.
    schedule.control    = struct('W', W2D, 'bc', bc2D);
    schedule.control(2) = struct('W', W2D, 'bc', bc2D);
    schedule.control(2).W.val = 0;
    
    % Specifying length of simulation timesteps
    schedule.step.val = [repmat(year/2,  20, 1); ...
                         repmat(10*year, 50, 1)];
    
    % Specifying which control to use for each timestep.
    % The first 20 timesteps will use control 1, the last 50
    % timesteps will use control 2.
    schedule.step.control = [ones(20, 1); ones(50, 1)*2];

end

