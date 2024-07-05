function[schedule] = gen_schedule(W, bc, fluid)

%R_inj = (1/num_wells) * 0.5 * 556.2 * 1000 * meter^3 / year;

num_wells = length(W);

cum_co2 = 0.5;
well_rate = cum_co2 * rand(num_wells, 20);
well_rate(well_rate<0.05) = 0;
well_rate = well_rate * (cum_co2*num_wells / sum(well_rate, 'all'));
well_rate = well_rate * 1e3 * mega / year / fluid.rhoGS;
well_rate = [well_rate, zeros(num_wells,10)];

for i=1:30
    schedule.control(i) = struct('W',W,'bc',bc);
    for k=1:num_wells
        schedule.control(i).W(k).val = well_rate(k,i);
    end
end

schedule.step.val = [repmat(year/2,20,1); repmat(100*year,10,1)];
schedule.step.control = linspace(1,30,30);

end

