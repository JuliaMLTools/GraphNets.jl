struct GNBlock
    edgefn
    nodefn
    graphfn
    dropout
end

Functors.@functor GNBlock

function GNBlock(from_to::Pair; dropout=0)
    # GNBlock(
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
    (; graphs, ef, nf, gf) = x
    h_ef = (m.edgefn ∘ getedgefninput)(graphs, ef, nf, gf)
    h_nf = (m.nodefn ∘ getnodefninput)(graphs, h_ef, nf, gf)
    h_gf = (m.graphfn ∘ getgraphfninput)(graphs, h_ef, h_nf, gf)
    (graphs=graphs, ef=h_ef, nf=h_nf, gf=h_gf)
end
