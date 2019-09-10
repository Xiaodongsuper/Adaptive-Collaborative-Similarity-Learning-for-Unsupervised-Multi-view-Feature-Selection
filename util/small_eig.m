% author:Xiao Dong 2018.1.17
function F = small_eig(value_F, k)
    [v d] = eig(value_F);
    d = diag(d);
    [d1, idx] = sort(d,'ascend');
    F = d(idx(1:k));