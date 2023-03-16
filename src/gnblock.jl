struct GNBlock
    edgefn
    nodefn
    graphfn
    dropout
end

Functors.@functor GNBlock

function GNBlock(from_to::Pair; dropout=0)
    # GNHead(
    #     Dense(input_dim, head_size, bias=false),
    #     Dense(input_dim, head_size, bias=false),
    #     Dense(input_dim, head_size, bias=false),
    #     Dropout(dropout),
    # )
    from, to = from_to
    @assert any(from .> (0,0,0))
    @assert any(to .> (0,0,0))
    edge_from, node_from, graph_from = from
    edge_to, node_to, graph_to = to
    edge_input_from = edge_from + 2*node_from + graph_from
    node_input_from = node_from + edge_to + graph_from
    graph_input_from = node_to + edge_to + graph_from
    GNBlock(
        Chain(Dense(edge_input_from, edge_to)),
        Chain(Dense(node_input_from, node_to)),
        Chain(Dense(graph_input_from, graph_to)),
        Dropout(dropout),
    )
end

function (m::GNBlock)(x)
    (graphs, edge_features, node_features, graph_features) = x
    uv, v, g = edge_features, node_features, graph_features
    edge_input = getedgefninput(
        graphs,
        edge_features,
        node_features,
        graph_features,
    )
    h_uv = (m.edgefn ∘ getedgefninput)(graphs, uv, v, g)
    h_u = (m.nodefn ∘ getnodefninput)(graphs, h_uv, v, g)
    h_g = (m.graphfn ∘ getgraphfninput)(graphs, h_uv, h_u, g)
    (h_uv, h_u, h_g)
end
