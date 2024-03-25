%% Trapping inventory
% The result is a more detailed inventory that accounts for six different categories of CO2:
%
% # Structural residual - CO2 residually trapped inside a structural trap
% # Residual - CO2 residually trapped outside any structural traps
% # Residual in plume - fraction of the CO2 plume outside any structural
%   traps that will be left behind as residually trapped droplets when the
%   plume migrates away from its current position
% # Structural plume - mobile CO2 volume that is currently contained within
%   a residual trap;  if the containing structure is breached, this volume
%   is free to migrate upward
% # Free plume - the fraction of the CO2 plume outside of structural traps
%   that is free to migrate upward and/or be displaced by imbibing brine.
% # Exited - volume of CO2 that has migrated out of the domain through its
%   lateral boundaries
%
% This model only has very small structural traps and residual trapping is
% therefore the main mechanism.

[Gt,~,transMult] = topSurfaceGrid(G);
rock2D           = averageRock(rock, Gt);

invPc3D = @(pc) (1-srw) .* (pe./max(pc, pe)).^2 + srw;
kr3D    = @(s) max((s-src)./(1-src), 0).^2; % uses CO2 saturation
VEfluid = makeVEFluid(Gt, rock, 'P-scaled table' , ...
               'co2_mu_ref'  , muco2             , ...
               'wat_mu_ref'  , muw               , ... 
               'co2_rho_ref' , rhoc              , ...
               'wat_rho_ref' , rhow              , ...
               'co2_rho_pvt' , [c_co2,   p_ref]  , ...
               'wat_rho_pvt' , [c_water, p_ref]  , ...
               'residual'    , [srw, src]        , ...
               'pvMult_p_ref', p_ref             , ...
               'pvMult_fac'  , c_rock            , ...
               'invPc3D'     , invPc3D           , ...
               'kr3D'        , kr3D              , ...
               'transMult'   , transMult);

ta      = trapAnalysis(Gt, false);
reports = makeReports(Gt, states, rock2D, VEfluid, schedule, [srw, src], ta, []);

h1 = figure; plot(1); ax = get(h1, 'currentaxes');
plotTrappingDistribution(ax, reports, 'legend_location', 'northwest');