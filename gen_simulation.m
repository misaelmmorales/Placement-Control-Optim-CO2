function [wellSol, states] = gen_simulation(G, rock, fluid, initState, schedule)
    %% Simulation
    lsolve = BackslashSolverAD('maxIterations', 500, ...
                                'tolerance', 1e-3);

    model = TwoPhaseWaterGasModel(G, rock, fluid, 0, 0);
    [wellSol, states] = simulateScheduleAD(initState, model, schedule, ...
                                           'LinearSolver', lsolve);

end