function [ta, reports] = gen_traps(Gt, states, model, schedule, props)

    % trap analysis
    ta = trapAnalysis(Gt, false);
    reports = makeReports(Gt, states, ...
                            model.rock, model.fluid, schedule,...
                            [props.srw, props.src], ta, []);

end