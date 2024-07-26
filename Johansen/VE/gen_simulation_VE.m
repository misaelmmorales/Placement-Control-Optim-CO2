function [wellSol,states] = gen_simulation_VE(Gt, rock2D, fluid, initState, schedule)

    lsolve = BackslashSolverAD('maxIterations', 500, 'tolerance', 1e-3);

    model             = CO2VEBlackOilTypeModel(Gt, rock2D, fluid);
    [wellSol, states] = simulateScheduleAD(initState, model, schedule);
    states = [{initState} states(:)'];

end

