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

h1 = figure; plot(1); ax = get(h1, 'currentaxes');
plotTrappingDistribution(ax, reports, 'legend_location', 'northwest');