function [W, WVE, wellIx] = gen_wells(CG, Gt, rock)
%GEN_WELLS Summary of this function goes here
%   Detailed explanation goes here

    num_wells = randi([1,5]);
    wellIx    = randi([8,32], [num_wells,2]);

    %R_inj = (1/num_wells) * 0.5 * 556.2 * 1000 * meter^3 / year; %0.5 MT/yr
    
    W = [];
    for i=1:num_wells
        W = verticalWell(W, CG, rock, wellIx(i,1), wellIx(i,2), 1,...
                         'InnerProduct' , 'ip_tpf' , ...
                         'Type'         , 'rate'      , ...
                         'Sign'         , 1           , ...
                         'comp_i'       , [1,0]       , ...
                         'Radius'       , 0.1         , ...
                         'refDepth'     , []           , ...
                         'name'         , ['I', int2str(i)]);
    end

    WVE = convertwellsVE(W, CG, Gt, rock);

end