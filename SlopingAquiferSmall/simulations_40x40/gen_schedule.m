function [schedule] = gen_schedule(tsteps, W2, bc, controls)
    
    % Extract time parameters:
    T          = tsteps(1);
    stopInject = tsteps(2);
    dTinj      = tsteps(3);
    dTmon      = tsteps(4);

    nTinj = stopInject/dTinj;
    nTmon = (T-stopInject)/dTmon;

    % adjust AD composition [W,G]
    for i=1:size(W2,1)
        W2(i).compi = [0,1];
    end

    % setup schedule steps
    schedule.step.val = [repmat(dTinj, nTinj, 1); repmat(dTmon, nTmon, 1)];
    schedule.step.control = [linspace(1,20,nTinj)'; repmat(21,nTmon,1)];

    % injection period
    for t=1:nTinj
        schedule.control(t) = struct('W',W2,'bc',bc);
        for i=1:size(W2,1)
            schedule.control(t).W(i).val = controls(i,t);
        end
    end

    % monitor period
    schedule.control(21) = struct('W',W2,'bc',bc);
    for i=1:size(W2,1)
        schedule.control(21).W(i).val = 0;
    end 

end