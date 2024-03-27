function [states] = simulation(i)
    %% Grid, Rock, and BCs
    grdecl = readGRDECL([fullfile(mrstPath('co2lab'),'data','johansen','NPD5'),'.grdecl']);
    % Construct grid structure.
    G = processGRDECL(grdecl);
    G = computeGeometry(G(1));
    % rock
    r = load(sprintf('data_100_100_11/rock/rock_%d.mat', i));
    p = r.poro(:);
    K = 10.^r.perm(:); 
    rock.perm = bsxfun(@times, [1 1 0.1], K(G.cells.indexMap)).*milli*darcy;
    rock.poro = p(G.cells.indexMap);
    clear p K;
    % boundary 3D
    nx = G.cartDims(1); ny=G.cartDims(2); nz=G.cartDims(3);
    ix1 = searchForBoundaryFaces(G, 'BACK', 1:nx-6, 1:4, 1:nz);
    ix2 = searchForBoundaryFaces(G, 'LEFT', 1:20,   1:ny, 1:nz);
    ix3 = searchForBoundaryFaces(G, 'RIGHT', 1:nx, ny-10:ny, 1:nz);
    ix4 = searchForBoundaryFaces(G, 'FRONT', 1:nx/2-8, ny/2:ny, 1:nz);
    bcIx = [ix1; ix2; ix3; ix4];

    %% Fluid
    gravity reset on;
    g       = gravity;
    co2     = CO2props();                             % CO2 property functions
    p_ref   = 30 *mega*Pascal;                        % reference pressure
    t_ref   = 94+273.15;                              % reference temperature
    rhow    = 1000;                                   % water density (kg/m^3)
    rhoc    = co2.rho(p_ref, t_ref);                  % CO2 density
    c_co2   = co2.rhoDP(p_ref, t_ref) / rhoc;         % CO2 compressibility
    c_water = 0;                                      % water compressibility
    c_rock  = 4.35e-5 / barsa;                        % rock compressibility
    srw     = 0.27;                                   % residual water
    src     = 0.20;                                   % residual CO2
    pe      = 5 * kilo * Pascal;                      % capillary entry pressure
    muw     = 8e-4 * Pascal * second;                 % brine viscosity
    muco2   = co2.mu(p_ref, t_ref) * Pascal * second; % co2 viscosity
    fluid = initSimpleADIFluid('phases', 'WG'           , ...
                               'mu'  , [muw, muco2]     , ...
                               'rho' , [rhow, rhoc]     , ...
                               'pRef', p_ref            , ...
                               'c'   , [c_water, c_co2] , ...
                               'cR'  , c_rock           , ...
                               'n'   , [2 2]);
    % relative permeability and capillary pressure
    fluid.krW  = @(s) fluid.krW(max((s-srw)./(1-srw), 0));
    fluid.krG  = @(s) fluid.krG(max((s-src)./(1-src), 0));
    pcWG       = @(sw) pe * sw.^(-1/2);
    fluid.pcWG = @(sg) pcWG(max((1-sg-srw)./(1-srw), 1e-5));
    % initial state
    initState.pressure = rhow * g(3) * G.cells.centroids(:,3);
    initState.s        = repmat([1, 0], G.cells.num, 1);
    initState.sGmax    = initState.s(:,2);
    % boundary conditions
    bc   = [];
    p_bc = G.faces.centroids(bcIx, 3) * rhow * g(3);
    bc   = addBC(bc, bcIx, 'pressure', p_bc, 'sat', [1,0]); 

    %% Well(s)
    inj_rate  = 3 * mega * 1e3 / year / rhoc;
    max_bhp   = []; %10000 * psia;
    num_wells = randi([1,3]);
    increment = 8072;
    actnum      = reshape(grdecl.ACTNUM, G.cartDims);
    actnum_l1   = actnum(:,:,1);
    well_loc_l1 = randsample(find(actnum_l1(:)), num_wells);
    well_locs = zeros(5,num_wells);
    for i=1:num_wells
        well_locs(:,i) = (well_loc_l1(i) + (0:4)*increment)';
    end
    W   = []; 
    for i=1:num_wells
       W = addWell(W, G, rock, well_locs(:,i), ...
                   'name'        , ['Injector', int2str(i)] , ...
                   'sign'        , 1                        , ...
                   'InnerProduct', 'ip_tpf'                 , ...
                   'type'        , 'rate'                   , ...
                   'val'         , inj_rate / num_wells     , ...
                   'lims'        , max_bhp                  , ...
                   'comp_i'      , [0 1]);
    end

    %% Schedule
    % total injection of 30 Mt CO2 over 10 years
    % change controls every 6 months (20 control steps)
    for i=1:20
        schedule.control(i) = struct('W', W, 'bc', bc);
    end
    schedule.control(21) = struct('W', W, 'bc', bc);
    min_rate  = 0.5             * mega * 1e3 / year / rhoc;
    well_rate = 10 * rand(1,20) * mega * 1e3 / year / rhoc;
    well_rate(well_rate<min_rate) = 0;
    well_rate = well_rate * (50 / (rhoc*sum(well_rate*year/2)/mega/1e3));
    for i=1:num_wells
        schedule.control(21).W(i).val = 0;
        for k=1:20
            schedule.control(k).W(i).val = well_rate(k);
        end
    end
    schedule.step.val     = [repmat(year/2,20,1); repmat(50*year,10,1)];
    schedule.step.control = [linspace(1,20,20)';  ones(10,1)*21];

    %% Run simulation
    model       = TwoPhaseWaterGasModel(G, rock, fluid, 0, 0);
    [~, states] = simulateScheduleAD(initState, model, schedule);
    states      = [{initState} states(:)'];
end