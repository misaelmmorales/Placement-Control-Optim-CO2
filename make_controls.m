function [VE_initState,VE_schedule] = make_controls(fluid, Gt, bcIxVE, W2D)
%MAKE_CONTROLS Summary of this function goes here

VE_initState.pressure = fluid.rhow*fluid.g(3)*Gt.cells.z;
VE_initState.s        = repmat([1,0], Gt.cells.num, 1);
VE_initState.sGmax    = VE_initState.s(:,2);

bc2D     = addBC([], bcIxVE, 'pressure', Gt.faces.z(bcIxVE) * fluid.rhow * fluid.g(3));
bc2D.sat = repmat([1,0], numel(bcIxVE), 1);

% schedule
min_rate  = 0.5             * mega * 1e3 / year / fluid.rhoc;
well_rate = 10 * rand(1,20) * mega * 1e3 / year / fluid.rhoc;
well_rate(well_rate<min_rate) = 0;
well_rate = well_rate * (50 / (fluid.rhoc*sum(well_rate*year/2)/mega/1e3));

for i=1:20
    VE_schedule.control(i) = struct('W', W2D, 'bc', bc2D);
end
VE_schedule.control(21) = struct('W', W2D, 'bc', bc2D);

num_wells = size(W2D,1);

for i=1:num_wells
    VE_schedule.control(21).W(i).val = 0;
    for k=1:20
        VE_schedule.control(k).W(i).val = well_rate(k);
    end
end

VE_schedule.step.val     = [repmat(year/2,20,1); repmat(50*year,10,1)];
VE_schedule.step.control = [linspace(1,20,20)';  ones(10,1)*21];

end

