function [VE_states, reports] = simulationVE(i)
    %% Grid, Rock, and BCs
    grdecl = readGRDECL([fullfile(mrstPath('co2lab'),'data','johansen','NPD5'),'.grdecl']);

    r = load(sprintf('data_100_100_11/rock/rock_%d.mat', i));
    p = r.poro(:);
    K = 10.^r.perm(:);  
    
    % Construct grid structure.
    G = processGRDECL(grdecl);
    G = computeGeometry(G(1));
    [Gt, G, transMult] = topSurfaceGrid(G);
    
    % Construct structure with petrophyiscal data.
    rock.perm = bsxfun(@times, [1 1 0.1], K(G.cells.indexMap)).*milli*darcy;
    rock.poro = p(G.cells.indexMap);
    rock2D    = averageRock(rock, Gt);
    clear p K;

    % boundary 2D
    nx = Gt.cartDims(1); ny=Gt.cartDims(2);
    ix1    = searchForBoundaryFaces(Gt, 'BACK',  1:nx-6, 1:4, []);
    ix2    = searchForBoundaryFaces(Gt, 'LEFT',  1:20, 1:ny,  []);
    ix3    = searchForBoundaryFaces(Gt, 'RIGHT', 1:nx, ny-10:ny, []);
    ix4    = searchForBoundaryFaces(Gt, 'FRONT', 1:nx/2-8, ny/2:ny, []);
    bcIxVE = [ix1; ix2; ix3; ix4];
    clear ix1 ix2 ix3 ix4 nx ny nz

    %% fluid
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

    %% Wells
    inj_rate    = 3 * mega * 1e3 / year / rhoc;
    max_bhp     = []; %10000 * psia;
    num_wells   = randi([1,3]);
    increment   = 8072;
    actnum      = reshape(grdecl.ACTNUM, G.cartDims);
    actnum_l1   = actnum(:,:,1);
    well_loc_l1 = randsample(find(actnum_l1(:)), num_wells);
    well_locs   = zeros(5, num_wells);
    for i=1:num_wells
       well_locs(:,i) = (well_loc_l1(i) + (0:4)*increment)';
    end
    W = [];
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
    W2D = convertwellsVE(W, G, Gt, rock2D);

    %% VE setup
    invPc3D  = @(pc) (1-srw) .* (pe./max(pc,pe)).^2 + srw; 
    kr3D     = @(s) max((s-src)./(1-src), 0).^2;
    VE_fluid = makeVEFluid(Gt, rock2D, 'P-scaled table'     , ...
                           'co2_mu_ref'  , muco2            , ...
                           'wat_mu_ref'  , muw              , ...
                           'co2_rho_ref' , rhoc             , ...
                           'wat_rho_ref' , rhow             , ...
                           'co2_rho_pvt' , [c_co2, p_ref]   , ...
                           'wat_rho_pvt' , [c_water, p_ref] , ...
                           'residual'    , [srw, src]       , ...
                           'pvMult_p_ref', p_ref            , ...
                           'pvMult_fac'  , c_rock           , ...
                           'invPc3D'     , invPc3D          , ...
                           'kr3D'        , kr3D             , ...
                           'transMult'   , transMult);
    
    VE_initState.pressure = rhow*g(3)*Gt.cells.z;
    VE_initState.s        = repmat([1,0], Gt.cells.num, 1);
    VE_initState.sGmax    = VE_initState.s(:,2);
        
    bc2D     = addBC([], bcIxVE, 'pressure', Gt.faces.z(bcIxVE)*rhow*g(3));
    bc2D.sat = repmat([1,0], numel(bcIxVE), 1);

    %% Schedule
    min_rate  = 0.5             * mega * 1e3 / year / rhoc;
    well_rate = 10 * rand(1,20) * mega * 1e3 / year / rhoc;
    well_rate(well_rate<min_rate) = 0;
    well_rate = well_rate * (100 / (rhoc*sum(well_rate*year/2)/mega/1e3));
    for i=1:20
        VE_schedule.control(i) = struct('W', W2D, 'bc', bc2D);
    end
    VE_schedule.control(21) = struct('W', W2D, 'bc', bc2D);
    for i=1:num_wells
        VE_schedule.control(21).W(i).val = 0;
        for k=1:20
            VE_schedule.control(k).W(i).val = well_rate(k);
        end
    end
    VE_schedule.step.val     = [repmat(year/2,20,1); repmat(50*year,10,1)];
    VE_schedule.step.control = [linspace(1,20,20)';  ones(10,1)*21];

    %% Simulation
    VE_model       = CO2VEBlackOilTypeModel(Gt, rock2D, VE_fluid);
    [~, VE_states] = simulateScheduleAD(VE_initState, VE_model, VE_schedule);
    VE_states      = [{VE_initState} VE_states(:)'];

    %% Trap Analysis
    ta      = trapAnalysis(Gt, false);
    reports = makeReports(Gt, VE_states, VE_model.rock, VE_model.fluid, VE_schedule, ...
                            [srw, src], ta, []);


    %% Plot
    %{
    figure(1); clf; plotCellData(G, rock.poro); plotWell(G,W); view(-63,50); colormap jet; colorbar
    figure(2); clf; plotCellData(G, log10(convertTo(rock.perm(:,1), milli*darcy))); plotWell(G,W); view(-63,50); colormap jet; colorbar
    figure(3); clf; plotToolbar(Gt, VE_states); view(-63,50); colormap(parula.^2)
    h1 = figure(4); plot(1); ax = get(h1, 'currentaxes');
    plotTrappingDistribution(ax, reports, 'legend_location', 'northwest');
    figure(5); clf; plot_VE_simulation
    %}

end