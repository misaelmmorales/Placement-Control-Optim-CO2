function [schedule] = gen_schedule_VE(W2D,bc2D,fluid)
    %% Schedule
    num_wells = length(W2D);

    cum_co2_inj = 1;
    well_rate = cum_co2_inj*rand(num_wells,20);
    well_rate(well_rate < 0.5) = 0;
    well_rate = well_rate * (cum_co2_inj*num_wells / sum(well_rate,'all'));
    well_rate = well_rate * 1e3 * mega / year / fluid.rhoGS;

    % 10 rampup timesteps
    rampsteps = rampupTimesteps(year/2, year/2, 9); 
    
    % 31 controls [10 rampup, 19 schedules, 10 monitor]
    for i=1:31
        schedule.control(i) = struct('W', W2D, 'bc', bc2D);
    end
    
    % update rates for each well
    for i=1:num_wells
        schedule.control(21).W(i).val = 0;
        for k=1:10
            schedule.control(k).W(i).val = well_rate(i,1);
        end

        for k=2:20
            schedule.control(k+9).W(i).val = well_rate(i,k);
        end
    end

    schedule.step.val     = [rampsteps;  repmat(year/2,19,1); repmat(100*year,10,1)];
    schedule.step.control = [ones(10,1); linspace(2,20,19)';  ones(10,1)*21];

end

