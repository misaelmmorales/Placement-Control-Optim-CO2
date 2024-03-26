function [W, W2D] = make_wells(G, Gt, grdecl, rock, rock2D, fluid, max_bhp)
% Setup the well(s)
    num_wells = randi([1,3]);
    increment = 8072;
    inj_rate  = 3 * mega * 1e3 / year / fluid.rhoc;

    actnum      = reshape(grdecl.ACTNUM, G.cartDims);
    actnum_l1   = actnum(:,:,1);
    well_loc_l1 = randsample(find(actnum_l1(:)), num_wells);

   well_locs = zeros(5, num_wells);
   for i=1:num_wells
       well_locs(:,i) = (well_loc_l1(i) + (0:4)*increment)';
   end

   W = [];
   for i=1:num_wells
       W = addWell(W, G, rock, well_locs(:,i), ...
                   'name', ['Injector', int2str(i)], ...
                   'sign', 1, ...
                   'InnerProduct', 'ip_tpf', ...
                   'type', 'rate', ...
                   'val', inj_rate / num_wells, ...
                   'lims', max_bhp, ...
                   'comp_i', [0 1]);
   end

   W2D = convertwellsVE(W, G, Gt, rock2D);
end