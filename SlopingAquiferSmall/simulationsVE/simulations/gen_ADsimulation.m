function [wellSol3d,states3d] = gen_ADsimulation(G, rock, fluid, initState, schedule)

    lsolve  = BackslashSolverAD('maxIterations',100,'tolerance',1e-3);
    csolve  = CPRSolverAD('maxIterations',100,'tolerance',1e-3);
    nlsolve = NonLinearSolver('useRelaxation'          , true , ...
                              'maxTimestepCuts'        , 10   , ...
                              'maxIterations'          , 25   , ...
                              'useLinesearch'          , true , ...
                              'LinearSolver'           , lsolve);
    
    model = TwoPhaseWaterGasModel(G, rock, fluid, 0, 0);

    [wellSol3d, states3d] = simulateScheduleAD(initState, model, schedule, ...
                                                'NonLinearSolver',nlsolve);

end

%{
'linesearchMaxIterations', 10, ...
%}