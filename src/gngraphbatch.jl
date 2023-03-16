struct GNGraphBatch
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


