cd(@__DIR__)
using GraphNets
using Functors
using Flux
using SparseArrays
using BenchmarkTools

struct GNModel
    node_embedding_table
    encoder
    core
    graph_layer_norm
    edge_head
    node_head
end

Functors.@functor GNModel

node_vocab_size = 100
input_dims = (0,256,0)
core_dims = (384,384,384)
output_dims = (2,2,0)

function GNModel(vocab_size, in_dims, core_dims, out_dims; n_core_blocks=2)
    _, node_in_size, _ = in_dims
    edge_core_size, node_core_size, _ = core_dims
    edge_out_size, node_out_size, _ = out_dims
    GNModel(
        Embedding(vocab_size=>node_in_size),
        GNBlock(in_dims, core_dims),
        GNCoreList([GNCore(core_dims; dropout=dropout) for _ in 1:n_core_blocks]),
        GraphLayerNorm(out_dims),
        Dense(edge_core_size=>edge_out_size),
        Dense(node_core_size=>node_out_size),
    )
end

function (m::GNModel)(graphs, idx, targets=nothing)
    featured_graphs = (
        graphs = graphs,
        ef = nothing,
        nf = m.node_embedding_table(idx),
        gf = nothing,
    )
    encoded = m.encoder(featured_graphs)
    x = m.core(encoded)
    x2_edge, x2_node, _ = m.graph_layer_norm(x) # (C,T,B)
    edge_logits = edge_head(x2_edge)
    node_logits = node_head(x2_node)
    if isnothing(targets)
        loss = nothing
    else

        # Stopped here.

        C, B, T = size(logits)
        logits_reshaped = reshape(logits, C, T*B)
        targets_reshaped = reshape(targets, T*B)
        targets_onehot = Flux.onehotbatch(targets_reshaped, 1:vocab_size)
        loss = Flux.logitcrossentropy(logits_reshaped, targets_onehot)
    end
    (logits=(edge_logits, node_logits), loss=loss)
end

struct GNGraphs
    padded_adj_mats
    node_counts
    edge_counts
    graphs
end

function GNGraphs(graphs)
    GNGraphs(
        padadjmats(graphs),
        getnodecounts(graphs),
        getedgecounts(graphs),
        graphs,
    )
end



function getsortededges(adj_mat)
    src, dst, _ = findnz(adj_mat)
    sort(collect(zip(src,dst)))
end

adj_mat = Flux.unsqueeze([1 0 1; 1 1 0; 0 0 1], dims=3)
idx = Flux.unsqueeze(repeat(collect(1:3), 1, 3), dims=3)
res = adj_mat .* idx
reduce(
    cat,
    
)

Flux.onehotbatch(res, 1:3)





dst = zeros(3,9,2)
for (slice_idx,slice) in enumerate(eachslice(masked_idx, dims=3))
    # @show slice_idx
    # @show size(slice[:])
    flat = slice[:]
    active_idx = findall(x->!iszero(x), flat)
    active = flat[active_idx]
    hot = Flux.onehotbatch(active, 1:3)
    # @show flat
    # @show active_idx
    # @show active
    # display(hot)
    NNlib.scatter!(+, view(dst, :, :, slice_idx), hot, active_idx)
end







sparse_adj_mats = [sparse(adj_mats[:,:,1]), sparse(adj_mats[:,:,2])]
sparse_adj_mats[1]

getnode2edgemat(adj_mats) |> size 
getnode2edgemat(adj_mats)


dst = zeros(3,9)
NNlib.scatter!(+, dst, [1 0 0; 0 1 0; 0 0 1], [3,2,1])




function getsortededges(adj_mat)
    src, dst, _ = findnz(adj_mat)
    sort(collect(zip(src,dst)))
end
function getedgesrcmask(adj_mat)
    sorted_edges = getsortededges(adj_mat)
    Float32.(
        reduce(
            hcat,
            map(first.(sorted_edges)) do idx
                Flux.onehot(idx, 1:3)
            end
        )
    )
end

T, E, B = 3, 9, 2
dst = zeros(T,E,B)
function getedgesrcmask(dst, slice_idx, adj_mat, labels)
    toscatter = Flux.onehotbatch(findnz(adj_mat)[1], labels)
    NNlib.scatter!(+, view(dst, :, :, slice_idx), toscatter, active_idx)
end

@btime getedgesrcmask(dst, 1, sparse_adj_mats[1], 1:3)