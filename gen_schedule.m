function [schedule] = gen_schedule(W, bc, fluid)
    %% Schedule
    num_wells = length(W);

    well_rate = 5*rand(num_wells,20);
    well_rate(well_rate < 0.5) = 0;
    well_rate = well_rate * (100*num_wells / sum(well_rate,'all'));
    well_rate = well_rate * 1e3 * mega / year / fluid.rhoGS;

    for i=1:20
        schedule.control(i) = struct('W', W, 'bc', bc);
    end
    schedule.control(21) = struct('W', W, 'bc', bc);

    for i=1:num_wells
        schedule.control(21).W(i).val = 0;
        for k=1:20
            schedule.control(k).W(i).val = well_rate(i,k);
        end
    end

    schedule.step.val     = [repmat(year/2,20,1); repmat(100*year,10,1)];
    schedule.step.control = [linspace(1,20,20)'; ones(10,1)*21];

end

