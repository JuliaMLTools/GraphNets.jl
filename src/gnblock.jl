struct GNBlock{E,N,G,D}
    edgefn::E
    nodefn::N
    graphfn::G
    dropout::D
end

Functors.@functor GNBlock

"""
    GNBlock(in => out;  dropout=0)

Initializes an instance of the **`GNBlock`** type, representing a GraphNet block.

The following keyword arguments are supported:
- `dropout` (Defaults to 0)

## Example:

```julia
in_dims = (X_DE, X_DN, X_DG) = 10, 5, 0
out_dims = (Y_DE, Y_DN, Y_DG) = 3, 4, 5
block = GNBlock(in_dims => out_dims)
adj_mat = adj_mat = [
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
    gf=graph_features # no input graph features
) |> batch
y = block(x) |> unbatch
@assert size(y.ef) == (Y_DE, num_edges, batch_size)
@assert size(y.nf) == (Y_DN, num_nodes, batch_size)
@assert size(y.gf) == (Y_DG, batch_size)
```
"""
function GNBlock((in, out)::Pair; dropout=0)
    @assert any(in .> (0,0,0))
    @assert any(out .> (0,0,0))
    edge_in, node_in, graph_in = in
    edge_out, node_out, graph_out = out
    edge_input_in = edge_in + 2*node_in + graph_in
    node_input_in = node_in + edge_out + graph_in
    graph_input_in = node_out + edge_out + graph_in
    GNBlock(
        Chain(Dense(edge_input_in, edge_out)),
        Chain(Dense(node_input_in, node_out)),
        Chain(Dense(graph_input_in, graph_out)),
        Dropout(dropout),
    )
end

function (m::GNBlock)(x)
    (; graphs, ef, nf, gf) = x
    h_ef = (m.edgefn ∘ getedgefninput)(graphs, ef, nf, gf)
    h_nf = (m.nodefn ∘ getnodefninput)(graphs, h_ef, nf, gf)
    h_gf = (m.graphfn ∘ getgraphfninput)(graphs, h_ef, h_nf, gf)
    (graphs=graphs, ef=h_ef, nf=h_nf, gf=h_gf)
end
