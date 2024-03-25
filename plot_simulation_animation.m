%% Animate the plume migration over the whole simulation period
figure

oG = generateCoarseGrid(Gt.parent, ones(Gt.parent.cells.num,1));
plotFaces(oG, 1:oG.faces.num,'FaceColor','none');
plotWell(Gt.parent, W,'FontSize',10)

view(-63, 50)
axis tight
colorbar
clim([0 1-srw]); colormap(parula.^2)

hs     = [];
time   = cumsum([0; schedule.step.val])/year;

if time < 10
    ptxt = 'Injection';
else
    ptxt = 'Migration';
end

for i=1:numel(states)
    delete(hs)
    [h, h_max] = upscaledSat2height(states{i}.s(:,2), states{i}.sGmax, Gt, ...
                                    'pcWG', fluid.pcWG, ...
                                    'rhoW', fluid.rhoW, ...
                                    'rhoG', fluid.rhoG, ...
                                    'p', states{end}.pressure);
    sat = height2Sat(struct('h', h, 'h_max', h_max), Gt, fluid);
    title(sprintf('Time: %4d yrs (%s)', time(i), ptxt));
    ix = sat>0; if ~any(ix), continue; end
    hs = plotCellData(Gt.parent, sat, ix); drawnow
end