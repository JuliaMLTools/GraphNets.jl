function getedgefninput(graphs, edge_features, node_features, graph_features)
    vcat(
        edge_features,
        NNlib.batched_mul(node_features, graphs.srcnode2edge_broadcaster),
        NNlib.batched_mul(node_features, graphs.dstnode2edge_broadcaster),
        NNlib.batched_mul(graph_features, graphs.graph2edge_broadcaster),
    )
end

function getedgefninput(graphs, edge_features, node_features, graph_features::Nothing)
    vcat(
        edge_features,
        NNlib.batched_mul(node_features, graphs.srcnode2edge_broadcaster),
        NNlib.batched_mul(node_features, graphs.dstnode2edge_broadcaster),
    )
end

function getedgefninput(graphs, edge_features, node_features::Nothing, graph_features)
    vcat(
        edge_features,
        NNlib.batched_mul(graph_features, graphs.graph2edge_broadcaster),
    )
end

function getedgefninput(graphs, edge_features, node_features::Nothing, graph_features::Nothing)
    edge_features
end

function getedgefninput(graphs, edge_features::Nothing, node_features, graph_features)
    vcat(
        NNlib.batched_mul(node_features, graphs.srcnode2edge_broadcaster),
        NNlib.batched_mul(node_features, graphs.dstnode2edge_broadcaster),
        NNlib.batched_mul(graph_features, graphs.graph2edge_broadcaster),
    )
end

function getedgefninput(graphs, edge_features::Nothing, node_features, graph_features::Nothing)
    vcat(
        NNlib.batched_mul(node_features, graphs.srcnode2edge_broadcaster),
        NNlib.batched_mul(node_features, graphs.dstnode2edge_broadcaster),
    )
end

function getedgefninput(graphs, edge_features::Nothing, node_features::Nothing, graph_features)
    vcat(
        NNlib.batched_mul(graph_features, graphs.graph2edge_broadcaster),
    )
end