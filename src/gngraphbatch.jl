struct GNGraphBatch
    adj_mats
    padded_adj_mats # (PN,PN,B)
    srcnode2edge_broadcaster # (PN, PN^2, B)
    dstnode2edge_broadcaster # (PN, PN^2, B)
    graph2edge_broadcaster # (1, PN^2, B)
    edge2node_broadcaster # (PN^2, PN, B)
    graph2node_broadcaster # (1, PN, B)
    edge2graph_broadcaster #
    node2graph_broadcaster #
end

Functors.@functor GNGraphBatch

function GNGraphBatch(adj_mats)
    padded_adj_mats = padadjmats(adj_mats)
    PN = size(padded_adj_mats, 1)
    GNGraphBatch(
        adj_mats,
        padded_adj_mats,
        getsrcnode2edgebroadcaster(padded_adj_mats),
        getdstnode2edgebroadcaster(padded_adj_mats),
        getgraph2edgebroadcaster(padded_adj_mats),
        getedge2nodebroadcaster(padded_adj_mats),
        getgraph2nodebroadcaster(padded_adj_mats),
        getedge2graphbroadcaster(adj_mats, PN),
        getnode2graphbroadcaster(adj_mats, PN),
    )
end

function getedge2graphbroadcaster(adj_mats, PN)
    B = length(adj_mats)
    PN2 = PN^2
    m = zeros(Float32, PN2, 1, B)
    for (graph_idx, adj_mat) in enumerate(adj_mats)
        N = size(adj_mat, 1)
        N2 = N^2
        m[1:N2, 1, graph_idx] .= adj_mat[:]
    end
    m
end

function getnode2graphbroadcaster(adj_mats, PN)
    B = length(adj_mats)
    m = zeros(Float32, PN, 1, B)
    for (graph_idx, adj_mat) in enumerate(adj_mats)
        N = size(adj_mat, 1)
        m[1:N, 1, graph_idx] .= 1
    end
    m
end

function getedge2nodebroadcaster(padded_adj_mats)
    PN,_,B = size(padded_adj_mats)
    PN2 = PN^2
    m = zeros(Float32, PN2, PN, B)
    for (batch_idx, padded_adj_mat) in enumerate(eachslice(padded_adj_mats, dims=3))
        for (col_idx,col) in enumerate(eachcol(padded_adj_mat))
            i_from = PN*(col_idx-1) + 1
            i_to = i_from + PN - 1
            m[i_from:i_to, col_idx, batch_idx] .= col
        end
    end
    m
end

function getgraph2nodebroadcaster(padded_adj_mats)
    PN,_,B = size(padded_adj_mats)
    ones(Float32, 1, PN, B)
end

function getgraph2edgebroadcaster(padded_adj_mats)
    PN,_,B = size(padded_adj_mats)
    PN2 = PN^2
    ones(Float32, 1, PN2, B)
end

getsrcnode2edgebroadcaster(padded_adj_mats) = getnode2edgebroadcaster(padded_adj_mats)
getdstnode2edgebroadcaster(padded_adj_mats) = getnode2edgebroadcaster(padded_adj_mats, transpose)

function getnode2edgebroadcaster(padded_adj_mats, src_dst_op=identity)
    PN,_,B = size(padded_adj_mats)
    PN2 = PN^2
    idx = repeat(1:PN, 1, PN) |> src_dst_op
    masked_idx = padded_adj_mats .* idx
    dst = zeros(Float32, PN,PN2,B)
    for (slice_idx,slice) in enumerate(eachslice(masked_idx, dims=3))
        flat = @view slice[:]
        active_idx = findall(x->!iszero(x), flat)
        active = @view flat[active_idx]
        hot = Flux.onehotbatch(active, 1:PN)
        NNlib.scatter!(+, view(dst, :, :, slice_idx), hot, active_idx)
    end
    dst
end


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



function batch(t::NamedTuple)
    @assert Set(keys(t)) == Set((:graphs, :ef, :nf, :gf))
    (; graphs, ef, nf, gf) = t
    @assert !isnothing(ef) || !isnothing(nf) || !isnothing(gf)
    batch(graphs,ef,nf,gf)
end

"""
    batch_size = 2
    edge_features = rand(Float32, X_DE, num_edges, batch_size)
    node_features = rand(Float32, X_DN, num_nodes, batch_size)
    graph_features = nothing # no graph level input features
    x = (
        graphs=adj_mat,  # All graphs in this batch have same structure
        ef=edge_features, 
        nf=node_features,
        gf=graph_features
    ) |> batch
"""
function batch(adj_mat::AbstractMatrix, ef, nf, gf)
    checkshapes3d(ef, nf, gf)
    checksamebatchsize3d(ef, nf, gf)
    checknodeedgecounts(adj_mat, ef, nf)
    num_nodes = size(adj_mat, 1)
    x = (
        graphs = GNGraphBatch([adj_mat]),
        ef = padef(adj_mat, ef),
        nf = nf, 
        gf = gf,
    )
end

"""
    edge_features = [
        rand(Float32, X_DE, num_edges_1),
        rand(Float32, X_DE, num_edges_2),
    ]
    node_features = [
        rand(Float32, X_DN, num_nodes_1),
        rand(Float32, X_DN, num_nodes_2),
    ]
    graph_features = nothing

    x = (
        graphs=[adj_mat_1,adj_mat_2],  # Graphs in this batch have different structure
        ef=edge_features, 
        nf=node_features,
        gf=graph_features
    ) |> batch
"""
function batch(adj_mats::AbstractVector, ef, nf, gf)
    @assert length(adj_mats) > 0
    checksamebatchsize2d(adj_mats, ef, nf, gf)

    # Check shapes
    for adj_mat in adj_mats
        @assert ndims(adj_mat) == 2 # (N, N)
    end
    checkshapes2d(ef, nf, gf)

    # Check node/edge counts
    for (adj_mat, ef_i, nf_i) in zip(adj_mats, ef, nf)
        checknodeedgecounts(adj_mat, ef_i, nf_i)
    end

    x = (
        graphs = GNGraphBatch(adj_mats),
        ef = batchef(adj_mats, ef),
        nf = batchnf(adj_mats, nf),
        gf = batchgf(gf),
    )
end

batchef(adj_mats, ef) = padef(adj_mats, ef)
batchef(adj_mats, ::Nothing) = nothing

batchnf(adj_mats, nf) = padnf(adj_mats, nf)
batchnf(adj_mats, ::Nothing) = nothing

batchgf(gf) = reduce(hcat, gf)
batchgf(::Nothing) = nothing

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

function unbatch(t::NamedTuple)
    @assert Set(keys(t)) == Set((:graphs, :ef, :nf, :gf))
    (; graphs, ef, nf, gf) = t
    @assert !isnothing(ef) || !isnothing(nf) || !isnothing(gf)
    unbatch(graphs,ef,nf,gf)
end

function unbatch(graphs::GNGraphBatch, ef, nf, gf)
    @assert !isnothing(ef) || !isnothing(nf) || !isnothing(gf)
    if length(graphs.adj_mats) == 1
        return unbatch(graphs.adj_mats[1], ef, nf, gf)
    end
    unbatch(graphs.adj_mats, ef, nf, gf)
end

function unbatch(adj_mat::AbstractMatrix, ef, nf, gf)
    @assert !isnothing(ef) || !isnothing(nf) || !isnothing(gf)
    (
        graphs=adj_mat,
        ef=unpadef(adj_mat, ef),
        nf=nf,
        gf=gf,
    )
end

function unbatch(adj_mats::AbstractVector, ef, nf, gf)
    @assert !isempty(ef) || !isempty(nf) || !isempty(gf)
    (
        graphs=adj_mats,
        ef=unpadef(adj_mats, ef),
        nf=unpadnf(adj_mats, nf),
        gf=gf,
    )
end

function unpadef(adj_mat::AbstractMatrix, ef)
    @view ef[:, findall(isone, view(adj_mat, :)), :]  
end

function unpadef(adj_mats::AbstractVector, ef)
    padded_adj_mats = padadjmats(adj_mats)
    map(enumerate(eachslice(padded_adj_mats; dims=3))) do (graph_idx,padded_adj_mat)
        idx = findall(isone, view(padded_adj_mat, :))
        view(ef, :, idx, graph_idx)
    end
end

function unpadnf(adj_mats::AbstractVector, nf)
    @show size(nf)
    map(enumerate(adj_mats)) do (graph_idx,adj_mat)
        @show size(adj_mat)
        @view nf[:, 1:size(adj_mat,1), graph_idx]
    end
end