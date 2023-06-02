function [newrock, rand_perm] = gen_monitorwells(G, rock, num_moni, loc_moni)
    %UNTITLED Summary of this function goes here
    %   Detailed explanation goes here
    
    %temp_poro = reshape(rock.poro, [G.cartDims]);
    %rand_poro = 10.^((log10(convertTo(rand_perm,milli*darcy))-7)/10);

    temp_perm = reshape(rock.perm(:,1), [G.cartDims]);
    mu        = mean(rock.perm(:,1));
    sigma     = std(rock.perm(:,1)) * randn([num_moni,1]);
    rand_perm = abs(mu+sigma);

    for i=1:num_moni
        %temp_poro(loc_moni(i,1), loc_moni(i,2), :) = rand_poro(i);
        temp_perm(loc_moni(i,1), loc_moni(i,2), :) = rand_perm(i);
    end

    %newrock.poro = reshape(temp_poro, [], 1);
    newrock.poro = rock.poro; %no change

    newperm = reshape(temp_perm, [], 1);
    newrock.perm(:,1) = newperm;
    newrock.perm(:,2) = newperm;
    newrock.perm(:,3) = 0.1*newperm;

end

