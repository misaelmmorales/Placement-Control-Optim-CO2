function [schedule, well_rate] = gen_schedule(W, bc, fluid)
    %% Schedule
    num_wells = length(W);
    
    cum_co2   = 15;
    well_rate = cum_co2*rand(num_wells,20);
    well_rate(well_rate < 2) = 0;
    well_rate = well_rate * (cum_co2*10*num_wells / sum(well_rate,'all'));
    well_rate = well_rate * 1e3 * mega / year / fluid.rhoGS;

    %%% [8 rampups, 20 injection, 5 monitor] (33 steps = 9+19+5)
    well_rate = [repmat(well_rate(:,1),1,8), well_rate, zeros(num_wells,5)];

    rampsteps = rampupTimesteps(10*year, year/2);

    for i=1:33
        schedule.control(i) = struct('W', W, 'bc', bc);
        for k=1:num_wells
            schedule.control(i).W(k).val = well_rate(k,i);
        end
    end

    schedule.step.val     = [rampsteps; repmat(20*year,5,1)];
    schedule.step.control = linspace(1,33,33);

end

%{
    %% Injection (10 years) + Monitoring (1000 years)
    cum_co2_inj = 1;
    well_rate = cum_co2_inj*rand(num_wells,20);
    well_rate(well_rate < 0.5) = 0;
    well_rate = well_rate * (cum_co2_inj*num_wells / sum(well_rate,'all'));
    well_rate = well_rate * 1e3 * mega / year / fluid.rhoGS;

    rampsteps = rampupTimesteps(year/2, year/2, 9); %10 rampups
    
    % 31 controls [10 rampup, 19 schedules, 10 monitor]
    for i=1:31
        schedule.control(i) = struct('W', W, 'bc', bc);
    end

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
%}