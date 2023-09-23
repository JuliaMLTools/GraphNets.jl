function padcat(v::AbstractVector)
    @assert (isone∘length∘unique)(size.(v, 1))
    D = size(first(v), 1)
    B = length(v)
    padded_block_size = maximum(size.(v, 2))
    m = zeros(Float32, D, padded_block_size, B)
    for (graph_idx,graph) in enumerate(v)
        block_size = size(graph, 2)
        m[1:D, 1:block_size, graph_idx]
    end
    m
end

function padmat(m::AbstractMatrix, pd1, pd2)
    @assert pd1 >= size(m,1)
    @assert pd2 >= size(m,2)
    d1, d2 = size(m)
    padded = zeros(Float32, pd1, pd2)
    padded[1:d1, 1:d2] .= m
    padded
end

function paddedbatch(v)
    @assert (isone∘length∘unique)(size.(v, 1))
    d1 = size(v[1], 1)
    d2 = maximum(size.(v, 2))
    padded = padmat.(v, d1, d2)
    reduce((a,b)->cat(a,b; dims=3), padded)
end

function getedgefeatures(t::NamedTuple, graph_idx)
    (; graphs, ef) = t
    getedgefeatures(graphs.adj_mats, graph_idx, ef)
end

function getnodefeatures(t::NamedTuple, graph_idx)
    (; graphs, nf) = t
    getnodefeatures(graphs.adj_mats, graph_idx, nf)
end

function getgraphfeatures(t::NamedTuple, graph_idx)
    (; graphs, gf) = t
    getgraphfeatures(graphs.adj_mats, graph_idx, gf)
end

function getedgefeatures(adj_mats, graph_idx, batched_ef)
    adj_mat = adj_mats[graph_idx]
    edge_idx = findall(isone,adj_mat[:])
    batched_ef[:,edge_idx,graph_idx]
end

function getnodefeatures(adj_mats, graph_idx, batched_nf)
    num_nodes = size(adj_mats[graph_idx], 1)
    batched_nf[:, 1:num_nodes, graph_idx]
end

function getgraphfeatures(adj_mats, graph_idx, batched_gf)
    batched_gf[:,1,graph_idx]
end

# Flux.Dense fix
# Note: please see the following for details:
# https://discourse.julialang.org/t/inconsistent-matrix-multiply-output-from-flux-dense-depending-on-shape-of-input/104116/10
# function (d::Dense)(x::AbstractArray)
#     Flux._size_check(d, x, 1 => size(d.weight, 2))
#     d.σ.(batched_mul(d.weight, x) .+ d.bias) 
# end