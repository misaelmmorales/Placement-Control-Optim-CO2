function [W, W2D, x, y] = gen_wells_VE(G, Gt, rock, rock2D)
    num_wells   = randi([1,5]);
    
    wc_global = false(G.cartDims);
    wc_global(G.cells.indexMap) = true;
    actnum_f = wc_global(:,:,end-1);
    index = find(actnum_f);
    select = index(randperm(length(index), num_wells));
    [x,y] = ind2sub(size(actnum_f), select);

    W = [];
    for i=1:num_wells
       W = verticalWell(W, G, rock, x(i), y(i), 6:10       , ...
                        'name'         , ['I', int2str(i)] , ...
                        'sign'         , 1                 , ...
                        'InnerProduct' , 'ip_tpf'          , ...
                        'type'         , 'rate'            , ...
                        'compi'        , [0 1]             );
    end

    W2D = convertwellsVE(W, G, Gt, rock2D);

end

