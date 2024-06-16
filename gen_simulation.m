function [wellSol, states] = gen_simulation(G, rock, fluid, initState, schedule)
    %% Simulation
    model = TwoPhaseWaterGasModel(G, rock, fluid, 0, 0);
    [wellSol, states] = simulateScheduleAD(initState, model, schedule);

end