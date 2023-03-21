import SnoopPrecompile

SnoopPrecompile.@precompile_all_calls begin
    X_DE = 10
    X_DN = 5
    X_DG = 2
    Y_DE = 3
    Y_DN = 4
    Y_DG = 5
    block = GNBlock(
        (X_DE,X_DN,X_DG) => (Y_DE,Y_DN,Y_DG)
    )
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
    graph_features = rand(Float32, X_DG, batch_size) 
    y = (
        graphs=adj_mat,
        ef=edge_features,
        nf=node_features,
        gf=graph_features,
    ) |> batch |> block |> unbatch
    adj_mat_1 = [
        1 0 1;
        1 1 0;
        0 0 1;
    ]
    num_nodes_1 = size(adj_mat_1, 1)
    num_edges_1 = length(filter(isone, adj_mat_1))
    adj_mat_2 = [
        1 0 1 0;
        1 1 0 1;
        0 0 1 0;
        1 1 0 1;
    ]
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
    graph_features = [
        rand(Float32, X_DG),
        rand(Float32, X_DG),
    ]
    y_batched = (
        graphs=[adj_mat_1,adj_mat_2],
        ef=edge_features, 
        nf=node_features,
        gf=graph_features
    ) |> batch |> block
    y = y_batched |> unbatch
    efview(y_batched, :, :, 1)
    nfview(y_batched, :, :, 1)
    gfview(y_batched, :, 1)
    efview(y_batched, :, :, 2)
    nfview(y_batched, :, :, 2)
    gfview(y_batched, :, 2)
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
    graph_features = rand(Float32, X_DG, batch_size) 
    y = (
        graphs=adj_mat,
        ef=edge_features,
        nf=node_features,
        gf=graph_features,
    ) |> batch |> block |> unbatch
end
