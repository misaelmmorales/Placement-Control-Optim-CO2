%% JOHANSEN BASE CASE
[G, Gt, bcIx, bcIxVE, ~, rock2D] = makeJohansenVEgrid();
rock = load('basecase_rock.mat').rock;
states = load('basecase_states.mat').states;
W = verticalWell([], G, rock,  48, 48, 6:10);

%% Grid
figure(1); clf; plotGrid(G); plotWell(G,W); view(-30,45); title('Grid')

%% Rock
figure(2); clf; plotCellData(G, rock.poro); 
colormap jet; colorbar; plotWell(G,W); view(-30,45); title('Porosity [v/v]')

figure(3); clf; plotCellData(G, log10(convertTo(rock.perm(:,1), milli*darcy)))
colormap jet; colorbar; plotWell(G,W); view(-30,45); title('LogPerm [mD]')

%% States
figure(3); clf; plotToolbar(G, states, 'edgecolor', 'k', 'edgealpha',0.5);
colormap jet; colorbar; plotWell(G,W); view(-30,45)