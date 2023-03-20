# GraphNets.jl

[![][docs-stable-img]][docs-stable-url]
[![][docs-dev-img]][docs-dev-url]

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://juliamltools.github.io/GraphNets.jl/dev/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://juliamltools.github.io/GraphNets.jl/stable/


## Example usage

```julia
using GraphNets

adj_mat = [
    1 0 1;
    1 1 0;
    0 0 1;
] # Adjacency matrix

num_nodes = length(adj_mat)
num_edges = length(filter(isone, adj_mat))

X_DE = 10 # Input feature dimension of edges
X_DN = 5 # Input feature dimension of nodes
X_DG = 0 # Input feature dimension of graphs (no graph level input data)
Y_DE = 3 # Output feature dimension of edges
Y_DN = 4 # Output feature dimension of nodes
Y_DG = 5 # Output feature dimension of graphs

block = GNBlock(
    (X_DE,X_DN,X_DG) => (Y_DE,Y_DN,Y_DG)
)

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

y = block(x) |> unbatch

@assert size(y.ef) == (Y_DE, num_edges, batch_size)
@assert size(y.nf) == (Y_DN, num_nodes, batch_size)
@assert size(y.gf) == (Y_DG, batch_size)


##########################################################
# Example #2: Batch of graphs with different structures
##########################################################

adj_mat_1 = [
    1 0 1;
    1 1 0;
    0 0 1;
] # Adjacency matrix 1
num_nodes_1 = length(adj_mat_1)
num_edges_1 = length(filter(isone, adj_mat_1))

adj_mat_2 = [
    1 0 1 0;
    1 1 0 1;
    0 0 1 0;
    1 1 0 1;
] # Adjacency matrix 2
num_nodes_2 = length(adj_mat_2)
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

y = block(x) |> unbatch

@assert size(y.ef[1]) == (Y_DE, num_edges_1)
@assert size(y.nf[1]) == (Y_DN, num_nodes_1)
@assert size(y.gf[1]) == (Y_DG,)

@assert size(y.ef[2]) == (Y_DE, num_edges_2)
@assert size(y.nf[2]) == (Y_DN, num_nodes_2)
@assert size(y.gf[2]) == (Y_DG,)
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