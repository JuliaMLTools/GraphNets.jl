struct GNCore
    block
    ffwd
    gn1
    gn2
end

Functors.@functor GNCore

"""
GNCore(dims; dropout=0)

Initializes an instance of the **`GNCore`** type, representing a GraphNet "core" block.

A **`GNCore`** instance accepts an input array **`x`** of dimensions (C, T, B) and outputs an array of dimensions (HS, T, B). "C" is the channel size (embedding dimension). "T" is the block size (number of input tokens). "B" is the batch size.

The following keyword arguments are supported:
- `dropout` (Defaults to 0)

## Examples:

```julia
dims = (DE, DN, DG) = 3, 4, 5
core = GNCore(dims)
adj_mat = adj_mat = [
    1 0 1;
    1 1 0;
    0 0 1;
]
num_nodes = size(adj_mat, 1)
num_edges = length(filter(isone, adj_mat))
batch_size = 2
edge_features = rand(Float32, DE, num_edges, batch_size)
node_features = rand(Float32, DN, num_nodes, batch_size)
graph_features = rand(Float32, DG, batch_size)
x = (
    graphs=adj_mat, # All graphs in this batch have same structure
    ef=edge_features, # (DE, num_edges, batch_size)
    nf=node_features, # (DN, num_nodes, batch_size)
    gf=graph_features # no input graph features
) |> batch
y = core(x) |> unbatch
@assert size(y.ef) == (DE, num_edges, batch_size)
@assert size(y.nf) == (DN, num_nodes, batch_size)
@assert size(y.gf) == (DG, batch_size)
```
"""
function GNCore(dims; dropout=0)
    @assert any(dims .> (0,0,0))
    GNCore(
        GNBlock(dims=>dims; dropout=dropout),
        GNFeedForward(dims; dropout=dropout),
        GNGraphNorm(dims),
        GNGraphNorm(dims),
    )
end

function (m::GNCore)(x)
    graphnetadd(graphnetadd(x, m.block(m.gn1(x))), m.ffwd(m.gn2(x)))
    # x + m.gnlayer(m.gn1(x)) + m.ffwd(m.gn2(x))
end

function graphnetadd(a, b)
    (
        graphs = a.graphs,
        ef = a.ef .+ b.ef,
        nf = a.nf .+ b.nf,
        gf = a.gf .+ b.gf,
    )
end