function [wellSol3d,states3d] = gen_ADsimulation(G, rock, fluid, initState, schedule)

    lsolve  = BackslashSolverAD('maxIterations',200,'tolerance',1e-3);
    csolve  = CPRSolverAD('maxIterations',500,'tolerance',1e-3);
    nlsolve = NonLinearSolver('useRelaxation'          , true , ...
                              'maxTimestepCuts'        , 10   , ...
                              'maxIterations'          , 100  , ...
                              'useLinesearch'          , true , ...
                              'LinearSolver'           , lsolve);
    
    model = TwoPhaseWaterGasModel(G, rock, fluid, 0, 0);

    [wellSol3d, states3d] = simulateScheduleAD(initState, model, schedule, ...
                                                'NonLinearSolver',nlsolve);

end

%{
'linesearchMaxIterations', 10, ...
%}