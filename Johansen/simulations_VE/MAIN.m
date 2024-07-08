mrstModule add ad-core ad-props co2lab
gravity reset on;

parfor i=1:1272
    [G, ~, bcIx, ~, ~, bcIxVE]           = makeJohansenVEgrid();
    [props]                              = gen_props();
    [rock]                               = gen_rock(i, G);
    [W]                                  = gen_well(G, rock, props);
    [Gt, transMult, rock2D, W2D, state0] = gen_2d(G, rock, W, props);
    [fluid]                              = gen_fluid(Gt, rock, props, transMult);
    [bc2D]                               = gen_bc(Gt, bcIxVE, props);
    [schedule]                           = gen_schedule(W2D, bc2D);
    [wellSol, states, model]             = gen_simulation(Gt, rock2D, fluid, state0, schedule);
    [ta, reports]                        = gen_traps(Gt, states, model, schedule, props);
    
end