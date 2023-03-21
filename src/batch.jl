"""
    batch(t::NamedTuple)

## Example:

```julia
dims = (DE, DN, DG) = 3, 4, 5
core = GNCore(dims)
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
    rand(Float32, DE, num_edges_1),
    rand(Float32, DE, num_edges_2),
]
node_features = [
    rand(Float32, DN, num_nodes_1),
    rand(Float32, DN, num_nodes_2),
]
graph_features = [
    rand(Float32, DG),
    rand(Float32, DG),
]
adj_mats = [adj_mat_1,adj_mat_2]
batch_size = length(adj_mats)
x = (
    graphs=adj_mats,  # Graphs in this batch have different structure
    ef=edge_features, 
    nf=node_features,
    gf=graph_features
)
x_batched = batch(x)
edge_block_size = x_batched.graphs.edge_block_size
node_block_size = x_batched.graphs.node_block_size
@assert typeof(x_batched.graphs) == GraphNets.GNGraphBatch
@assert size(x_batched.ef) == (DE, edge_block_size, batch_size)
@assert size(x_batched.nf) == (DN, node_block_size, batch_size)
@assert size(x_batched.gf) == (DG, 1, batch_size)
```
"""
function batch(t::NamedTuple)
    @assert Set(keys(t)) == Set((:graphs, :ef, :nf, :gf))
    (; graphs, ef, nf, gf) = t
    @assert !isnothing(ef) || !isnothing(nf) || !isnothing(gf)
    checks(graphs,ef,nf,gf)
    (
        graphs = batchgraphs(graphs),
        ef = batchef(graphs, ef),
        nf = batchnf(graphs, nf), 
        gf = batchgf(graphs, gf),
    )
end

batchgraphs(adj_mat::AbstractMatrix) = GNGraphBatch([adj_mat])
batchgraphs(adj_mats::AbstractVector) = GNGraphBatch(adj_mats)

batchef(adj_mats, ef) = padef(adj_mats, ef)
batchef(adj_mats, ::Nothing) = nothing

batchnf(adj_mats, nf) = padnf(adj_mats, nf)
batchnf(adj_mats, ::Nothing) = nothing

batchgf(adj_mats, gf) = padgf(adj_mats, gf)
batchgf(adj_mats, ::Nothing) = nothing