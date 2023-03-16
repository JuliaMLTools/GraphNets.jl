struct GNGraphBatch
    padded_adj_mats # (PN,PN,B)
    src_node_broadcaster # (PN, PN^2, B)
    dst_node_broadcaster # (PN, PN^2, B)
    graph_broadcaster # (1, PN^2, B)
    edge2node_broadcaster
    graph2node_broadcaster
end

Functors.@functor GNGraphBatch

function GNGraphBatch(adj_mats)
    padded_adj_mats = padadjmats(adj_mats)
    GNGraphBatch(
        padded_adj_mats,
        getsrcnodebroadcaster(padded_adj_mats),
        getdstnodebroadcaster(padded_adj_mats),
        getgraphbroadcaster(padded_adj_mats),
        edge2nodebroadcaster(padded_adj_mats),
        graph2nodebroadcaster(padded_adj_mats),
    )
end

function edge2nodebroadcaster(padded_adj_mats)
end

function graph2nodebroadcaster(padded_adj_mats)
end

function getgraphbroadcaster(padded_adj_mats)
    PN,_,B = size(padded_adj_mats)
    PN2 = PN^2
    ones(1, PN2, B)
end

getsrcnodebroadcaster(padded_adj_mats) = getnodebroadcaster(padded_adj_mats)
getdstnodebroadcaster(padded_adj_mats) = getnodebroadcaster(padded_adj_mats, transpose)

function getnodebroadcaster(padded_adj_mats, src_dst_op=identity)
    PN,_,B = size(padded_adj_mats)
    PN2 = PN^2
    idx = repeat(1:PN, 1, PN) |> src_dst_op
    masked_idx = padded_adj_mats .* idx
    dst = zeros(PN,PN2,B)
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


