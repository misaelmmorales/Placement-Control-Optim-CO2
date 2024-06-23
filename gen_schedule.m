function [schedule] = gen_schedule(W, bc, fluid)
    %% Schedule
    num_wells = length(W);
    
    cum_co2 = 20;
    well_rate = cum_co2*rand(num_wells,20);
    well_rate(well_rate < 1) = 0;
    well_rate = well_rate * (cum_co2*10*num_wells / sum(well_rate,'all'));
    well_rate = well_rate * 1e3 * mega / year / fluid.rhoGS;

    rampsteps = rampupTimesteps(10*year, year/2);

    for i=1:28
        schedule.control(i) = struct('W', W, 'bc', bc);
    end
    
    for k=1:num_wells
        for i=1:8
            schedule.control(i).W(k).val = well_rate(k,1);
        end
        for i=9:28
            schedule.control(i).W(k).val = well_rate(k,i-8);
        end
    end

    schedule.step.val     = rampsteps;
    schedule.step.control = [ones(9,1); linspace(2,20,19)'];

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