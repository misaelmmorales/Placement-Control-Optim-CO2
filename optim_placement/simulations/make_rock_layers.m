function [poro, perm] = make_rock_layers(startlayer,endlayer)

    rock = getSPE10rock(startlayer:endlayer);
    nz = endlayer - startlayer + 1;
    poro = reshape(rock.poro, [60,220, nz]);
    perm = reshape(rock.perm(:,1), [60, 220, nz]);

end

