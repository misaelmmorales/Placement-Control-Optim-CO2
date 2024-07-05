function [bc2D] = gen_bc(Gt, bcIxVE, props)

    % hydrostatic pressure conditions for open boundary faces
    p_bc     = Gt.faces.z(bcIxVE) * props.rhow * props.g(3);
    bc2D     = addBC([], bcIxVE, 'pressure', p_bc); 
    bc2D.sat = repmat([1 0], numel(bcIxVE), 1);

end