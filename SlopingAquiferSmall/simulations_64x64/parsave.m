function parsave(fname, var)
%PARSAVE save variable within a parfor loop
    save(fname, 'var')
end