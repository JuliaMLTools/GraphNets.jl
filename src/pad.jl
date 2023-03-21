function padadjmats(adj_mats)
    B = length(adj_mats)
    PN = maximum(first.(size.(adj_mats)))
    batched = zeros(Float32, PN, PN, B)
    for (idx,adj_mat) in enumerate(adj_mats)
        N = first(size(adj_mat))
        batched[1:N, 1:N, idx] .= adj_mat
    end
    batched
end

padnf(::AbstractMatrix, nf) = nf

function padnf(adj_mats::AbstractVector, nfs::AbstractVector)
    max_num_nodes = maximum(first.(size.(adj_mats)))
    node_dim = size(nfs[1], 1)
    batch_size = length(adj_mats)
    padded_nf = zeros(Float32, node_dim, max_num_nodes, batch_size)
    for (graph_idx,nf) in enumerate(nfs)
        N = size(nf, 2)
        padded_nf[:, 1:N, graph_idx] .= nf
    end
    padded_nf
end

function padef(adj_mat::AbstractMatrix, ef)
    batch_size = size(ef, 3)
    num_nodes = size(adj_mat, 1)
    N2 = num_nodes^2
    edge_idx = findall(isone, view(adj_mat, :))
    edge_dim = size(ef, 1)
    padded = zeros(Float32, edge_dim, N2, batch_size)
    for (graph_idx,slice) in enumerate(eachslice(ef; dims=3))
        NNlib.scatter!(+, view(padded, :, :, graph_idx), slice, edge_idx)
    end
    padded
end

function padef(adj_mat::AbstractMatrix, ef::AbstractMatrix)
    edge_dim = size(ef, 1)
    num_nodes = size(adj_mat, 1)
    N2 = num_nodes^2
    edge_idx = findall(isone, view(adj_mat, :))
    padded = zeros(Float32, edge_dim, N2)
    NNlib.scatter!(+, padded, ef, edge_idx)
end

function padef(adj_mats::AbstractVector, efs::AbstractVector)
    padded_adj_mats = padadjmats(adj_mats)
    reduce(
        (a,b)->cat(a,b,dims=3),
        map(
            pair->begin
                padded_adj_mat, ef_i = pair
                padef(padded_adj_mat, ef_i)
            end,
            zip(eachslice(padded_adj_mats; dims=3), efs),
        )
    )
end

padgf(adj_mats::AbstractMatrix, gf) = reshape(gf, :, 1, size(gf, 2)) # (D, 1, B)
padgf(adj_mats::AbstractVector, gf) = reshape(reduce(hcat, gf), :, 1, length(gf)) # (D, 1, B)