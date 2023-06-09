cd(@__DIR__)
import Pkg
Pkg.activate(".")
include("imports.jl")
Random.seed!(1234)

##########################
# Create a sample graph with input/output features
##########################
vocab_size = 100 # Maximum integer to be sorted

function gensample()
    n = rand(2:10) # number of nodes
    adj_mat = ones(Int, n, n) # fully connected graph
    num_edges = length(filter(isone, adj_mat))
    x_nf = rand(1:100, n) # Sample input node features
    y_nf = Int.(x_nf .== minimum(x_nf)) # Sample output node features
    y_ef = getedgetargets(x_nf) # Output edge features
    (
        adj_mat=adj_mat, 
        x=(; nf=Flux.onehotbatch(x_nf, 1:vocab_size)),
        y=(nf=Flux.onehotbatch(y_nf, 0:1), ef=Flux.onehotbatch(y_ef, 0:1))
    )
end

###########################
# Training batch generator
###########################
device = CUDA.functional() ? gpu : cpu

function getbatch(batch_size)
    samples = [gensample() for _ in 1:batch_size]
    x = (
        graphs=[s.adj_mat for s in samples],
        ef=nothing, 
        nf=[s.x.nf for s in samples],
        gf=nothing
    ) |> batch |> device
    target = (
        graphs=[s.adj_mat for s in samples],
        ef=[s.y.ef for s in samples],
        nf=[s.y.nf for s in samples],
        gf=nothing,
    ) |> batch |> device
    x, target
end

##########################
# Define the GraphNet GNN
##########################
struct GNModel
    encoder
    core
    decoder
end

Functors.@functor GNModel

function GNModel(from_to::Pair, core_dims; n_core_blocks=2)
    x_dims, y_dims = from_to
    _, x_dn, _ = x_dims
    GNModel(
        GNBlock(x_dims => core_dims),
        GNCoreList([GNCore(core_dims) for _ in 1:n_core_blocks]),
        GNBlock(core_dims => y_dims),
    )
end

function (m::GNModel)(xs, targets=nothing)
    encoded = m.encoder(xs)
    x1 = m.core(encoded)
    ŷ = m.decoder(x1)
    if isnothing(targets)
        loss = nothing
    else
        loss_nodes = Flux.logitcrossentropy(flatunpaddednf(ŷ), flatunpaddednf(targets))
        loss_edges = Flux.logitcrossentropy(flatunpaddedef(ŷ), flatunpaddedef(targets))
        loss = loss_nodes + loss_edges
    end
    (graph=ŷ, loss=loss)
end

##########################
# Init untrained model
##########################
x_dims = (0,vocab_size,0)
core_dims = (384,384,384)
y_dims = (2,2,0)
model = GNModel(x_dims=>y_dims, core_dims) |> device

##########################
# Ensure model runs on a batch
##########################
x, y = getbatch(2)
y_batched, loss = model(x,y)

##########################
# Show a training sample and the output
##########################
function showsample(model)
    cpu_model = model |> cpu
    x, y = getbatch(1) .|> cpu
    ŷ_batched, loss = cpu_model(x, y)
    ŷ = ŷ_batched |> unbatch
    x = SVG([SVGText("X"), getinputgraph(x)])
    y = SVG([SVGText("Y"), gettargetgraph(y)])
    ŷ = SVG([SVGText("Ŷ"), gettargetgraph(ŷ)])
    SVG([x,y,ŷ], dims=2)
end

showsample(model)

##########################
# Setup hyperparameters and train
##########################
batch_size = 4
learning_rate = 3e-4
optim = Flux.setup(Flux.AdamW(learning_rate), model)
dropout = 0
max_iters = 20_000

function train!(model)
    trainmode!(model)
    @showprogress for iter in 1:max_iters
        xb, yb = getbatch(batch_size)
        loss, grads = Flux.withgradient(model) do m
            m(xb, yb).loss
        end
        Flux.update!(optim, model, grads[1])
    end
    testmode!(model)
end

train!(model)

##########################
# Check results
##########################
showsample(model)