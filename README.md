# GraphNets.jl

[![][docs-stable-img]][docs-stable-url]
[![][docs-dev-img]][docs-dev-url]

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://juliamltools.github.io/GraphNets.jl/dev/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://juliamltools.github.io/GraphNets.jl/stable/


## Example usage

```julia
X_DE = 10 # Input feature dimension of edges
X_DN = 5 # Input feature dimension of nodes
X_DG = 0 # Input feature dimension of graphs (no graph level input data)
Y_DE = 3 # Output feature dimension of edges
Y_DN = 4 # Output feature dimension of nodes
Y_DG = 5 # Output feature dimension of graphs

block = GNBlock(
    (X_DE,X_DN,X_DG) => (Y_DE,Y_DN,Y_DG)
)


##########################################################
# Example #1: Batch of graphs with same structure (same adjacency matrix)
##########################################################

adj_mat = [
    1 0 1;
    1 1 0;
    0 0 1;
] # Adjacency matrix

num_nodes = size(adj_mat, 1)
num_edges = length(filter(isone, adj_mat))

batch_size = 2
edge_features = rand(Float32, X_DE, num_edges, batch_size)
node_features = rand(Float32, X_DN, num_nodes, batch_size)
graph_features = nothing # no graph level input features

x = (
    graphs=adj_mat, # All graphs in this batch have same structure
    ef=edge_features, # (X_DE, num_edges, batch_size)
    nf=node_features, # (X_DN, num_nodes, batch_size)
    gf=graph_features # (X_DG, batch_size)
) |> batch

y = block(x) |> unbatch

@assert size(y.ef) == (Y_DE, num_edges, batch_size)
@assert size(y.nf) == (Y_DN, num_nodes, batch_size)
@assert size(y.gf) == (Y_DG, batch_size)

# Get the output graph edges of the 1st graph
@assert size(y.ef[:,:,1]) == (Y_DE, num_edges)

# Get the output node edges of the 1st graph
@assert size(y.nf[:,:,1]) == (Y_DN, num_nodes)

# Get the output graph edges of the 2nd graph
@assert size(y.gf[:,2]) == (Y_DG,)


##########################################################
# Example #2: Batch of graphs with different structures
##########################################################

adj_mat_1 = [
    1 0 1;
    1 1 0;
    0 0 1;
] # Adjacency matrix 1
num_nodes_1 = size(adj_mat_1, 1)
num_edges_1 = length(filter(isone, adj_mat_1))

adj_mat_2 = [
    1 0 1 0;
    1 1 0 1;
    0 0 1 0;
    1 1 0 1;
] # Adjacency matrix 2
num_nodes_2 = size(adj_mat_2, 1)
num_edges_2 = length(filter(isone, adj_mat_2))

edge_features = [
    rand(Float32, X_DE, num_edges_1),
    rand(Float32, X_DE, num_edges_2),
]
node_features = [
    rand(Float32, X_DN, num_nodes_1),
    rand(Float32, X_DN, num_nodes_2),
]
graph_features = nothing # no graph level input features

x = (
    graphs=[adj_mat_1,adj_mat_2],  # Graphs in this batch have different structure
    ef=edge_features, 
    nf=node_features,
    gf=graph_features
) |> batch

y_batched = block(x)
y = y_batched |> unbatch

# Memory-efficient view of features for a batch with different graph structures
@assert size(efview(y_batched, :, :, 1)) == (Y_DE, num_edges_1) # edge features for graph 1
@assert size(nfview(y_batched, :, :, 1)) == (Y_DN, num_nodes_1)  # edge features for graph 1
@assert size(gfview(y_batched, :, 1)) == (Y_DG,) # graph features for graph 1
@assert size(efview(y_batched, :, :, 2)) == (Y_DE, num_edges_2) # edge features for graph 2
@assert size(nfview(y_batched, :, :, 2)) == (Y_DN, num_nodes_2) # node features for graph 2
@assert size(gfview(y_batched, :, 2)) == (Y_DG,) # graph features for graph 2

# Copied array of features (less efficient) for a batch with different graph structures
@assert size(y.ef[1]) == (Y_DE, num_edges_1) # edge features for graph 1
@assert size(y.nf[1]) == (Y_DN, num_nodes_1)  # edge features for graph 1
@assert size(y.gf[1]) == (Y_DG,) # graph features for graph 1
@assert size(y.ef[2]) == (Y_DE, num_edges_2) # edge features for graph 2
@assert size(y.nf[2]) == (Y_DN, num_nodes_2) # node features for graph 2
@assert size(y.gf[2]) == (Y_DG,) # graph features for graph 2


####
# Example #3: Sequential GraphNet blocks
####

input_dims = (X_DE, X_DN, X_DG)
core_dims = (10, 5, 3)
output_dims = (Y_DE, Y_DN, Y_DG)

struct GNNModel{E,C,D}
    encoder::E
    core_list::C
    decoder::D
end

function GNNModel(; n_cores=2)
    GNNModel(
        GNBlock(input_dims => core_dims),
        GNCoreList([GNCore(core_dims) for _ in 1:n_cores]),
        GNBlock(core_dims => output_dims),
    )
end

function (m::GNNModel)(x)
    (m.decoder ∘ m.core_list ∘ m.encoder)(x)
end

m = GNNModel()

adj_mat = [
    1 0 1;
    1 1 0;
    0 0 1;
]

num_nodes = size(adj_mat, 1)
num_edges = length(filter(isone, adj_mat))

batch_size = 2
edge_features = rand(Float32, X_DE, num_edges, batch_size)
node_features = rand(Float32, X_DN, num_nodes, batch_size)
graph_features = nothing # no graph level input features

x = (
    graphs=adj_mat, # All graphs in this batch have same structure
    ef=edge_features, # (X_DE, num_edges, batch_size)
    nf=node_features, # (X_DN, num_nodes, batch_size)
    gf=graph_features # (X_DG, batch_size)
) |> batch

y = block(x) |> unbatch

@assert size(y.ef) == (Y_DE, num_edges, batch_size)
@assert size(y.nf) == (Y_DN, num_nodes, batch_size)
@assert size(y.gf) == (Y_DG, batch_size)
```


## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add GraphNets
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("GraphNets")
```

## Project Status

The package is tested against, and being developed for, Julia `1.8` and above on Linux, macOS, and Windows.