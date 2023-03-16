function getnodefninput(graphs, edge_features, node_features, graph_features)
    vcat(
        NNlib.batched_mul(edge_features, graphs.edge2node_broadcaster),
        node_features,
        NNlib.batched_mul(graph_features, graphs.graph2node_broadcaster),
    )
end

function getnodefninput(graphs, edge_features, node_features, graph_features::Nothing)
    vcat(
        NNlib.batched_mul(edge_features, graphs.edge2node_broadcaster),
        node_features,
    )
end

function getnodefninput(graphs, edge_features, node_features::Nothing, graph_features)
    vcat(
        NNlib.batched_mul(edge_features, graphs.edge2node_broadcaster),
        NNlib.batched_mul(graph_features, graphs.graph2node_broadcaster),
    )
end

function getnodefninput(graphs, edge_features, node_features::Nothing, graph_features::Nothing)
    NNlib.batched_mul(edge_features, graphs.edge2node_broadcaster)
end