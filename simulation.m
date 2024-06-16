function [states, wellSol] = simulation(realization, bc_type)
    %% Grid, Rock, and BCs
    [G, Gt, bcIx] = makeJohansenVEgrid('verbose',0);

    r = load(sprintf('data_100_100_11/rock/rock_%d.mat', realization));
    p = r.poro(:);
    K = 10.^r.perm(:);  
    
    % Construct structure with petrophyiscal data.
    rock.perm = bsxfun(@times, [1 1 0.1], K(G.cells.indexMap)).*milli*darcy;
    rock.poro = p(G.cells.indexMap);
    clear p K;

    %% Initial State
    gravity on;
    g = gravity;
    rhow = 1000;
    initState.pressure = rhow * g(3) * G.cells.centroids(:,3);
    initState.s = repmat([1, 0], G.cells.num, 1);
    initState.sGmax = initState.s(:,2);

    %% Fluid model
    co2     = CO2props();
    p_ref   = 30 * mega * Pascal;
    t_ref   = 94 + 273.15;
    rhoc    = co2.rho(p_ref, t_ref);
    cf_co2  = co2.rhoDP(p_ref, t_ref) / rhoc;
    cf_wat  = 0;
    cf_rock = 4.35e-5 / barsa;
    muw     = 8e-4 * Pascal * second;
    muco2   = co2.mu(p_ref, t_ref) * Pascal * second;
    fluid = initSimpleADIFluid('phases', 'WG'           , ...
                               'mu'  , [muw, muco2]     , ...
                               'rho' , [rhow, rhoc]     , ...
                               'pRef', p_ref            , ...
                               'c'   , [cf_wat, cf_co2] , ...
                               'cR'  , cf_rock          , ...
                               'n'   , [2 2]);
    srw = 0.27;
    src = 0.20;
    fluid.krW = @(s) fluid.krW(max((s-srw)./(1-srw), 0));
    fluid.krG = @(s) fluid.krG(max((s-src)./(1-src), 0));
    pe = 5 * kilo * Pascal;
    pcWG = @(sw) pe * sw.^(-1/2);
    fluid.pcWG = @(sg) pcWG(max((1-sg-srw)./(1-srw), 1e-5)); 

    %% Boundary conditions
    bc = [];    
    p_bc = G.faces.centroids(bcIx, 3) * rhow * g(3);    
    if strcmp(bc_type, 'noflow')
        bc = addBC(bc, bcIx, 'flux', 0, 'sat', [1,0]);
    elseif strcmp(bc_type, 'constP')
        bc = addBC(bc, bcIx, 'pressure', p_bc, 'sat', [1, 0]);
    else
        error('Invalid BC (bc_type). Choose either (noflow) or (constP)')
    end

    %% Well(s)
    inj_rate  = 5 * mega * 1e3 / year / rhoc;
    max_bhp   = 5000*psia;

    num_wells = randi([1,5]);

    wc_global = false(G.cartDims);
    wc_global(G.cells.indexMap) = true;
    actnum_f = wc_global(:,:,end-1);
    index = find(actnum_f);
    select = index(randperm(length(index), num_wells));
    [x,y] = ind2sub(size(actnum_f), select);

    W = [];
    for i=1:num_wells
       W = verticalWell(W, G, rock, x(i), y(i), 7:10, ...
                        'name'         , ['Injector', int2str(i)] , ...
                        'sign'         , 1                        , ...
                        'InnerProduct' , 'ip_tpf'                 , ...
                        'type'         , 'rate'                   , ...
                        'val'          , inj_rate / num_wells     , ...
                        'lims'         , max_bhp                  , ...
                        'comp_i'       , [0 1]);
    end

    %% Schedule
    total_co2_inj = 100/num_wells;
    well_rate = 5*rand(num_wells,20);
    well_rate(well_rate < 0.5) = 0;
    
    rowSum = sum(well_rate,2);
    scalingFactors = total_co2_inj ./ rowSum;
    for i=1:num_wells
        well_rate(i,:) = well_rate(i,:) * scalingFactors(i);
    end

    well_rate = well_rate * 1e3 * mega / year / fluid.rhoGS;

    for i=1:20
        schedule.control(i) = struct('W', W, 'bc', bc);
    end
    schedule.control(21) = struct('W', W, 'bc', bc);

    for i=1:num_wells
        schedule.control(21).W(i).val = 0;
        for k=1:20
            schedule.control(k).W(i).val = well_rate(i,k);
        end
    end

    schedule.step.val     = [repmat(year/2,20,1); repmat(50*year,20,1)];
    schedule.step.control = [linspace(1,20,20)'; ones(20,1)*21];

    %% Simulation
    model = TwoPhaseWaterGasModel(G, rock, fluid, 0, 0);
    [wellSol, states] = simulateScheduleAD(initState, model, schedule);

end