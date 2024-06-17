function [schedule] = gen_schedule(W, bc, fluid)
    %% Schedule
    num_wells = length(W);

    well_rate = 5*rand(num_wells,20);
    well_rate(well_rate < 0.5) = 0;
    well_rate = well_rate * (7.5*10*num_wells / sum(sum(well_rate)));
    well_rate = well_rate * 1e3 * mega / year / fluid.rhoGS;

    rampsteps = rampupTimesteps(year/2, year/2, 9);

    for i=1:30
        schedule.control(i) = struct('W', W, 'bc', bc);
    end
    schedule.control(31) = struct('W', W, 'bc', bc);

    for i=1:num_wells
        schedule.control(31).W(i).val = 0;
        for k=1:10
            schedule.control(k).W(i).val = well_rate(i,1);
        end
        for k=2:10
            schedule.control(k+9).W(i).val = well_rate(i,k);
        end
    end

    schedule.step.val     = [rampsteps;  repmat(year/2,19,1); repmat(100*year,10,1)];
    schedule.step.control = [ones(10,1); linspace(2,20,19)';  ones(10,1)*21];

end

