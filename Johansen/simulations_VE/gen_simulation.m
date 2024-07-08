function [wellSol, states, model] = gen_simulation(Gt, rock2D, fluid, initState, schedule)
    % Create and simulate model
    model = CO2VEBlackOilTypeModel(Gt, rock2D, fluid);
    [wellSol, states] = simulateScheduleAD(initState, model, schedule);
    states = [{initState} states(:)'];
end

