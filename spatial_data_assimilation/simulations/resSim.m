function [saturation] = resSim(perm_field)

  G = cartGrid([51,51,1], [4000*meter, 4000*meter, 100*meter]);
  G = computeGeometry(G);

  logperm   = perm_field';
  permx     = 10.^logperm*milli*darcy;
  permz     = 0.1*permx;
  perm      = [permx, permx, permz];
  %poro      = 10.^((logperm-7)/10);        %Kozeny-Carman porosity
  poro      = repmat(0.15, G.cells.num, 1); %constant porosity
  rock.poro = poro;
  rock.perm = perm;

  gravity on;
  P0 = 4000*psia;
  initState.pressure = repmat(P0, G.cells.num, 1);
  initState.s        = repmat([1, 0], G.cells.num, 1);
  initState.sGmax    = initState.s(:,2);

  co2     = CO2props();                     % load sampled tables of co2 fluid properties
  p_ref   = 30 * mega * Pascal;             % choose reference pressure
  t_ref   = 94 + 273.15;                    % choose reference temperature, in Kelvin
  rhow    = 1000;                           % density of brine at ref press/temp
  rhoc    = co2.rho(p_ref, t_ref);          % co2 density at ref. press/temp
  cf_co2  = co2.rhoDP(p_ref, t_ref) / rhoc; % co2 compressibility
  cf_wat  = 0;                              % brine compressibility (zero)
  cf_rock = 4.35e-5 / barsa;                % rock compressibility
  muw     = 8e-4 * Pascal * second;         % brine viscosity
  muco2   = co2.mu(p_ref, t_ref) * Pascal * second; % co2 viscosity
  fluid   = initSimpleADIFluid('phases', 'WG'           , ...
                               'mu'  , [muw, muco2]     , ...
                               'rho' , [rhow, rhoc]     , ...
                               'pRef', p_ref            , ...
                               'c'   , [cf_wat, cf_co2] , ...
                               'cR'  , cf_rock          , ...
                               'n'   , [2 2]);
  [srw,src]  = deal(0.27,0.20);
  fluid.krW  = @(s) fluid.krW(max((s-srw)./(1-srw), 0));
  fluid.krG  = @(s) fluid.krG(max((s-src)./(1-src), 0));
  pe         = 5 * kilo * Pascal;
  pcWG       = @(sw) pe * sw.^(-1/2);
  fluid.pcWG = @(sg) pcWG(max((1-sg-srw)./(1-srw), 1e-5));

  bc = [];
  vface_ind       = (G.faces.normals(:,3) == 0);
  bface_ind       = (prod(G.faces.neighbors, 2) == 0);
  bc_face_ix      = find(vface_ind & bface_ind);
  bc_cell_ix      = sum(G.faces.neighbors(bc_face_ix, :), 2);
  p_face_pressure = initState.pressure(bc_cell_ix);
  bc              = addBC(bc, bc_face_ix, ...
                          'pressure', p_face_pressure, ...
                          'sat',      [1,0]);

  total_time = 5*year;
  timestep   = rampupTimesteps(total_time, year, 5);
  %timestep = [0.25 0.25 0.5 2 2]*year;
  irate      = (1/3)*sum(poreVolume(G, rock))/total_time;

  W = [];
  W = verticalWell(W, G, rock, 25, 25, [] ,...
                  'Type',         'rate'  , ...
                  'Val',          irate   , ...
                  'InnerProduct', 'ip_tpf', ...
                  'Comp_i',       [0 1]   , ...
                  'name',         'Injector');

  schedule.control      = struct('W', W, 'bc', bc);
  schedule.step.val     = timestep;
  schedule.step.control = ones(numel(timestep),1);

  model       = TwoPhaseWaterGasModel(G, rock, fluid);
  [~, states] = simulateScheduleAD(initState, model, schedule);

  sat = zeros(length(timestep), G.cells.num);
  for i=1:length(timestep)
    sat(i,:) = states{i,1}.s(:,2);
  end

  saturation = sat([6,8,10],:);
  %saturation = sat;
  
end