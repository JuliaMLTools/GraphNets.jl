function getgraphfninput(graphs, edge_features, node_features, graph_features)
    vcat(
        NNlib.batched_mul(edge_features, graphs.edge2graph_broadcaster),
        NNlib.batched_mul(node_features, graphs.node2graph_broadcaster),
        graph_features,
    )
end
function getgraphfninput(graphs, edge_features, node_features, graph_features::Nothing)
    vcat(
        NNlib.batched_mul(edge_features, graphs.edge2graph_broadcaster),
        NNlib.batched_mul(node_features, graphs.node2graph_broadcaster),
    )
end