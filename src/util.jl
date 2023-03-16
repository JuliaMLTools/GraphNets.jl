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

function getedgefeatures(t::Tuple, graph_idx)
    graphs, ef, _, _ = t
    getedgefeatures(graphs.adj_mats, graph_idx, ef)
end

function getnodefeatures(t::Tuple, graph_idx)
    graphs, _, nf, _ = t
    getnodefeatures(graphs.adj_mats, graph_idx, nf)
end

function getgraphfeatures(t::Tuple, graph_idx)
    graphs, _, _, gf = t
    getgraphfeatures(graphs.adj_mats, graph_idx, gf)
end

function getedgefeatures(adj_mats, graph_idx, batched_ef)
    adj_mat = adj_mats[graph_idx]
    edge_idx = findall(isone,adj_mat[:])
    batched_ef[:,edge_idx,graph_idx]
end

function getnodefeatures(adj_mats, graph_idx, batched_nf)
    batched_nf[:,:,graph_idx]
end

function getgraphfeatures(adj_mats, graph_idx, batched_gf)
    batched_gf[:,1,graph_idx]
end