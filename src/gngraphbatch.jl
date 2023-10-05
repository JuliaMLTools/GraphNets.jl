struct GNGraphBatch{A,P,SN2E,DN2E,G2E,E2N,G2N,E2G,N2G,EB,NB,FNU,FEU,EC,CEI}
    adj_mats::A
    padded_adj_mats::P # (PN,PN,B)
    srcnode2edge_broadcaster::SN2E # (PN, PN^2, B)
    dstnode2edge_broadcaster::DN2E # (PN, PN^2, B)
    graph2edge_broadcaster::G2E # (1, PN^2, B)
    edge2node_broadcaster::E2N # (PN^2, PN, B)
    graph2node_broadcaster::G2N # (1, PN, B)
    edge2graph_broadcaster::E2G #
    node2graph_broadcaster::N2G #
    edge_block_size::EB
    node_block_size::NB
    flat_node_unpadder::FNU
    flat_edge_unpadder::FEU
    edge_collapser::EC
    collapse_edge_idxs::CEI
end

Functors.@functor GNGraphBatch (
    srcnode2edge_broadcaster, 
    dstnode2edge_broadcaster,
    graph2edge_broadcaster,
    edge2node_broadcaster,
    graph2node_broadcaster,
    edge2graph_broadcaster,
    node2graph_broadcaster,
    flat_node_unpadder,
    flat_edge_unpadder,
    edge_collapser,
    collapse_edge_idxs,
)

function GNGraphBatch(adj_mats)
    padded_adj_mats = padadjmats(adj_mats)
    node_block_size = size(padded_adj_mats, 1)
    edge_block_size = node_block_size^2
    GNGraphBatch(
        adj_mats,
        padded_adj_mats,
        getsrcnode2edgebroadcaster(padded_adj_mats),
        getdstnode2edgebroadcaster(padded_adj_mats),
        getgraph2edgebroadcaster(padded_adj_mats, edge_block_size),
        getedge2nodebroadcaster(padded_adj_mats),
        getgraph2nodebroadcaster(adj_mats, node_block_size),
        getedge2graphbroadcaster(padded_adj_mats, node_block_size),
        getnode2graphbroadcaster(adj_mats, node_block_size),
        edge_block_size,
        node_block_size,
        getflatnodeunpadder(adj_mats, node_block_size),
        getflatedgeunpadder(adj_mats, edge_block_size),
        getedgecollapser(node_block_size),
        getcollapsededgeidxs(padded_adj_mats),
    )
end

function getlowertriangularcoords(m)
    filter(idx -> idx[1] >= idx[2], CartesianIndices(m))
end

function getcollapsededgeidxs(padded_adj_mats)
    map(eachslice(padded_adj_mats, dims=3)) do padded_adj_mat
        lower_tri_coords = getlowertriangularcoords(padded_adj_mat)
        findall(isone, padded_adj_mat[lower_tri_coords])
    end
end

function getedgecollapser(node_block_size)
    n = node_block_size
    m = reshape(1:(n^2), n, n)
    lower_tri_coords = getlowertriangularcoords(m)
    reduce(
        hcat,
        map(lower_tri_coords) do coord
            i,j = coord[1],coord[2]
            copy = zeros(eltype(m), size(m))
            copy[i,j] += 1
            copy[j,i] += 1
            view(copy, :)
        end
    )
end

function collapsef(graph::NamedTuple)
    batched_mul(graph.ef, graph.graphs.edge_collapser) / Float32(2)
end

function unpaddedcollapsedef(graph::NamedTuple)
    unpaddedcollapsedef(
        collapsef(graph), 
        graph.graphs.collapse_edge_idxs,
    )
end

function unpaddedcollapsedef(collapsed_efs, collapsed_idxs::Vector{<:Integer})
    map(eachslice(collapsed_efs, dims=3)) do collapsed_ef
        @view collapsed_ef[:, collapsed_idxs]
    end
end

function unpaddedcollapsedef(collapsed_efs, collapsed_idxs_list::Vector{<:Vector})
    map(zip(
            eachslice(collapsed_efs, dims=3), 
            collapsed_idxs_list,
        )) do (collapsed_ef, collapsed_idxs)
        @view collapsed_ef[:, collapsed_idxs]
    end
end

function flatunpaddedcollapsedef(graph::NamedTuple)
    reduce(hcat, unpaddedcollapsedef(graph))
end

function getflatnodeunpadder(adj_mats, PN)
    B = length(adj_mats)
    mask = zeros(Bool, B*PN)
    for (i, adj_mat) in enumerate(adj_mats)
        num_nodes = size(adj_mat, 1)
        s = 1 + PN*(i-1)
        e = s + num_nodes - 1
        mask[s:e] .= 1
    end
    mask
end

function getflatedgeunpadder(adj_mats, PE)
    B = length(adj_mats)
    mask = zeros(Bool, B*PE)
    for (i, adj_mat) in enumerate(adj_mats)
        s = 1 + PE*(i-1)
        e = s + length(adj_mat) - 1
        mask[s:e] .= view(adj_mat, :)
    end
    mask
end

function getedge2graphbroadcaster(padded_adj_mats, PN)
    B = size(padded_adj_mats, 3)
    PN2 = PN^2
    m = zeros(Float32, PN2, 1, B)
    for (graph_idx, padded_adj_mat) in enumerate(eachslice(padded_adj_mats, dims=3))
        N = size(padded_adj_mat, 1)
        N2 = N^2
        m[1:N2, 1, graph_idx] .= view(padded_adj_mat, :)
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

function getgraph2nodebroadcaster(adj_mats, node_block_size)
    reduce(
        (a,b)->cat(a,b; dims=3), 
        map(adj_mats) do adj_mat
            m = zeros(Float32, 1, node_block_size)
            m[1:size(adj_mat,1)] .= 1
            m
        end
    )
end

function getgraph2edgebroadcaster(padded_adj_mats, edge_block_size)
    reduce(
        (a,b)->cat(a,b; dims=3), 
        map(eachslice(padded_adj_mats, dims=3)) do padded_adj_mat
            m = zeros(Float32, 1, edge_block_size)
            m[findall(isone, reshape(padded_adj_mat, :))] .= 1
            m
        end
    )
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