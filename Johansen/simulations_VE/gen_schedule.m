function [schedule, well_rate] = gen_schedule(W2D, bc2D, props)
    %% Schedule
    num_wells = length(W2D);
    
    cum_co2   = 10;
    well_rate = cum_co2*rand(num_wells,20);
    well_rate(well_rate < 2) = 0;
    well_rate = well_rate * (cum_co2*10*num_wells / sum(well_rate,'all'));
    well_rate = well_rate * 1e3 * mega / year / props.co2_rho;

    %%% [8 rampups, 20 injection, 5 monitor] (33 steps = 9+19+5)
    well_rate = [repmat(well_rate(:,1),1,8), well_rate, zeros(num_wells,5)];

    rampsteps = rampupTimesteps(10*year, year/2);

    for i=1:33
        schedule.control(i) = struct('W', W2D, 'bc', bc2D);
        for k=1:num_wells
            schedule.control(i).W(k).val = well_rate(k,i);
        end
    end

    schedule.step.val     = [rampsteps; repmat(20*year,5,1)];
    schedule.step.control = linspace(1,33,33);

end

