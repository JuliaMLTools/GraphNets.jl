cd(@__DIR__)
include("imports.jl")

g = Graphs.SimpleGraphs.SimpleGraph(ones(10,10))
gplot(g)

struct GNModel
    node_embedding_table
    encoder
    core
    decoder
end

Functors.@functor GNModel

function GNModel(from_to::Pair, core_dims, vocab_size; n_core_blocks=2)
    x_dims, y_dims = from_to
    _, x_dn, _ = x_dims
    GNModel(
        Embedding(vocab_size => x_dn),
        GNBlock(x_dims => core_dims),
        GNCoreList([GNCore(core_dims) for _ in 1:n_core_blocks]),
        GNBlock(core_dims => y_dims),
    )
end

function (m::GNModel)(graphs, idx, targets=nothing)
    x = (
        graphs = graphs,
        ef = nothing,
        nf = m.node_embedding_table(idx.nodes),
        gf = nothing,
    )
    encoded = m.encoder(x)
    x1 = m.core(encoded)
    y = m.decoder(x1)
    logits = (
        edge_logits = y.ef,
        node_logits = y.nf,
    )
    if isnothing(targets)
        loss = nothing
    else
        DN, T, B = size(logits.node_logits)
        
        node_logits_reshaped = reshape(logits.node_logits, DN*T*B)
        node_targets_reshaped = reshape(targets.nodes, T*B)
        loss_nodes = Flux.logitbinarycrossentropy(node_logits_reshaped, node_targets_reshaped)

        DE, T, B = size(logits.edge_logits)
        edge_logits_reshaped = reshape(logits.edge_logits, DE*T*B)
        edge_targets_reshaped = reshape(targets.edges, T*B)
        loss_edges = Flux.logitbinarycrossentropy(edge_logits_reshaped, edge_targets_reshaped)
        
        loss = loss_nodes + loss_edges
    end
    (logits=logits, loss=loss)
end


# batch_size = 64 # how many independent sequences will we process in parallel?
# block_size = 256 # what is the maximum context length for predictions?
# max_iters = 2000
# eval_interval = 500
# learning_rate = 3e-4
# eval_iters = 200
# n_embd = 384
# n_head = 6
# head_size = n_embd ÷ n_head
# inv_sqrt_dₖ = Float32(1 / sqrt(head_size))
# n_layer = 6
# dropout = 0.2
# device = CUDA.functional() ? gpu : cpu

data = rand(1:100, 20_000)
n = Int(round(0.9*length(data))) # first 90% will be train, rest val
train_data = data[1:n]
val_data = data[n+1:end]

block_size = 10
batch_size = 2

function getedgetargets(node_idx)
    n = length(node_idx)

    node_idx_idx = collect(zip(1:n, node_idx))
    sorted = first.(sort(node_idx_idx; lt=(a,b)->last(a) < last(b)))
    enabled_edges = collect(zip(sorted[1:end-1], sorted[2:end]))
    # @show node_idx
    # @show node_idx_idx
    # @show sorted
    # @show enabled_edges
    edge_targets_mat = zeros(Int, n, n)
    for (i,j) in enabled_edges
        edge_targets_mat[i,j] = 1
    end
    edge_targets = edge_targets_mat[:]
    # display(edge_targets_mat)

    edge_targets
end

function getbatch(split)
    # generate a small batch of data of inputs x and targets y
    data = split == "train" ? train_data : val_data
    ix = rand(1:(length(data) - block_size), batch_size)
    x_nodes = reduce(hcat, [data[i:i+block_size-1] for i in ix])
    x_nodes_sorted = reduce(hcat, map(sort, eachcol(x_nodes)))
    x_nodes_min = reduce(hcat, map(minimum, eachcol(x_nodes_sorted)))
    y_nodes = broadcast(==, x_nodes_min, x_nodes)
    y_edges = reduce(hcat, getedgetargets.(eachcol(x_nodes)))
    (
        (; nodes=x_nodes),
        (nodes=y_nodes, edges=y_edges),
    )
end
getbatch("train")

Random.seed!(1234)

vocab_size = 100 # Maximum integer to be sorted

# Let's overfit the network on a single example to start
N = 10
x_idx = rand(1:vocab_size, N, 2) # Unsorted integers
y_idx = sort(x_idx) # Sorted integers
adj_mat = ones(N, N) # Fully connected graph
adj_mats = [adj_mat] # Just one graph in the batch
G = length(adj_mats) # number of graphs

x_dims = (0,256,0)
core_dims = (384,384,384)
y_dims = (1,1,0)

model = GNModel(x_dims=>y_dims, core_dims, vocab_size)
logits, loss = model(GNGraphBatch(adj_mats), x_idx)
logits.edge_logits
logits.node_logits

x, y = getbatch("train")
logits, loss = model(GNGraphBatch(adj_mats), x, y)
model(GNGraphBatch(adj_mats), x).logits.node_logits

learning_rate = 3e-4
optim = Flux.setup(Flux.AdamW(learning_rate), model)
dropout = 0
max_iters = 10_000

function train!(model)
    graphs = GNGraphBatch(adj_mats)
    trainmode!(model)
    @showprogress for iter in 1:max_iters
        xb, yb = getbatch("train")
        loss, grads = Flux.withgradient(model) do m
            m(graphs, xb, yb).loss
        end
        Flux.update!(optim, model, grads[1])
    end
    testmode!(model)
end

train!(model)

x, y = getbatch("valid")
logits, loss = model(GNGraphBatch(adj_mats), x, y)
reshape(sigmoid(model(GNGraphBatch(adj_mats), x).logits.node_logits), 10, 2)
reshape(Int.(round.(sigmoid(model(GNGraphBatch(adj_mats), x).logits.edge_logits))), 10, 10, 2)