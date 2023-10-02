using GraphNets
using Test

@testset "Test edge collapsing" begin  
    in_dims = (X_DE, X_DN, X_DG) = (0, 2, 0)
    core_dims = (2,2,2)
    out_dims = (Y_DE, Y_DN, Y_DG) = (2,2,2)
    
    encoder = GNBlock(in_dims => core_dims)
    decoder = GNBlock(core_dims => out_dims)

    block(x) = (decoder ∘ encoder)(x)

    adj_mat_A  = [
        1 1;
        1 1;
    ]
    adj_mat_B = [
        1 1 1;
        1 1 1;
        1 1 1;
    ]
    num_nodes_A = size(adj_mat_A, 1)
    num_edges_A = length(filter(isone, adj_mat_A))
    num_nodes_B = size(adj_mat_B, 1)
    num_edges_B = length(filter(isone, adj_mat_B))
    batch_size = 2
    
    node_features = [rand(Float32, X_DN, num_nodes_A), rand(Float32, X_DN, num_nodes_B)]

    x_nbatch = (
        graphs=[adj_mat_A, adj_mat_B], # All graphs in this batch have same structure
        ef=nothing,
        nf=node_features[:],
        gf=nothing # no input graph features
    ) |> batch
    
    ŷ_nbatch = block(x_nbatch)
    collapsed_flat = flatunpaddedcollapsedef(ŷ_nbatch)
    
    @test all(collapsed_flat[:,1] .≈ ŷ_nbatch.ef[:,1,1])
    @test all(collapsed_flat[:,2] .≈ (ŷ_nbatch.ef[:,2,1] + ŷ_nbatch.ef[:,4,1])/2)
    @test all(collapsed_flat[:,3] .≈ ŷ_nbatch.ef[:,5,1])

    @test all(collapsed_flat[:,4] .≈ ŷ_nbatch.ef[:,1,2])
    @test all(collapsed_flat[:,5] .≈ (ŷ_nbatch.ef[:,2,2] + ŷ_nbatch.ef[:,4,2])/2)
    @test all(collapsed_flat[:,6] .≈ (ŷ_nbatch.ef[:,3,2] + ŷ_nbatch.ef[:,7,2])/2)
    @test all(collapsed_flat[:,7] .≈ ŷ_nbatch.ef[:,5,2])
    @test all(collapsed_flat[:,8] .≈ (ŷ_nbatch.ef[:,6,2] + ŷ_nbatch.ef[:,8,2])/2)
    @test all(collapsed_flat[:,9] .≈ ŷ_nbatch.ef[:,9,2])

    # (
    #     graph=ŷ_nbatch,
    #     ef=ŷ_nbatch.ef,
    #     collapser=ŷ_nbatch.graphs.edge_collapser,
    #     collapsed_ef=collapsef(ŷ_nbatch),
    #     collapsed_flat=collapsed_flat,
    # )
end
# (; graph, ef, collapser, collapsed_ef, collapsed_flat) = testcollapsedeges()

@testset "GNBlock batch invariance" begin    
    in_dims = (X_DE, X_DN, X_DG) = (0, 2, 0)
    core_dims = (2,2,2)
    out_dims = (Y_DE, Y_DN, Y_DG) = (2,2,2)
    
    encoder = GNBlock(in_dims => core_dims)
    decoder = GNBlock(core_dims => out_dims)

    block(x) = (decoder ∘ encoder)(x)

    adj_mat_A  = [
        1 1;
        1 1;
    ]
    adj_mat_B = [
        1 1 1;
        1 1 1;
        1 1 1;
    ]
    num_nodes_A = size(adj_mat_A, 1)
    num_edges_A = length(filter(isone, adj_mat_A))
    num_nodes_B = size(adj_mat_B, 1)
    num_edges_B = length(filter(isone, adj_mat_B))
    batch_size = 2
    
    node_features = [rand(Float32, X_DN, num_nodes_A), rand(Float32, X_DN, num_nodes_B)]

    x_1batch = (
        graphs=[adj_mat_A], # All graphs in this batch have same structure
        ef=nothing,
        nf=node_features[1:1],
        gf=nothing # no input graph features
    ) |> batch
    ŷ_1batch = block(x_1batch)
    nf_1batch_1 = nfview(ŷ_1batch, :, :, 1)
    ef_1batch_1 = efview(ŷ_1batch, :, :, 1)
    gf_1batch_1 = gfview(ŷ_1batch, :, 1)

    x_nbatch = (
        graphs=[adj_mat_A, adj_mat_B], # All graphs in this batch have same structure
        ef=nothing,
        nf=node_features[:],
        gf=nothing # no input graph features
    ) |> batch
    ŷ_nbatch = block(x_nbatch)
    nf_nbatch_1 = nfview(ŷ_nbatch, :, :, 1)
    ef_nbatch_1 = efview(ŷ_nbatch, :, :, 1)
    gf_nbatch_1 = gfview(ŷ_nbatch, :, 1)

    @test size(nf_1batch_1) == size(nf_nbatch_1)
    @test size(ef_1batch_1) == size(ef_nbatch_1)
    @test all(nf_1batch_1 .≈ nf_nbatch_1)
    @test all(ef_1batch_1 .≈ ef_nbatch_1)
    @test all(gf_1batch_1 .≈ gf_nbatch_1)
end

@testset "no graph features output" begin

    # Setup
    X_DE = 10 # Input feature dimension of edges
    X_DN = 5 # Input feature dimension of nodes
    X_DG = 0 # Input feature dimension of graphs (no graph level input data)
    Y_DE = 3 # Output feature dimension of edges
    Y_DN = 4 # Output feature dimension of nodes
    Y_DG = 0 # Output feature dimension of graphs
    
    block = GNBlock(
        (X_DE,X_DN,X_DG) => (Y_DE,Y_DN,Y_DG)
    )
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

    @test size(y.ef) == (Y_DE, num_edges, batch_size)
    @test size(y.nf) == (Y_DN, num_nodes, batch_size)
    @test isnothing(y.gf)

    # Get the output graph edges of the 1st graph
    @test size(y.ef[:,:,1]) == (Y_DE, num_edges)

    # Get the output node edges of the 1st graph
    @test size(y.nf[:,:,1]) == (Y_DN, num_nodes)
    
end

@testset verbose = true "Readme Examples" begin

    # Setup
    X_DE = 10 # Input feature dimension of edges
    X_DN = 5 # Input feature dimension of nodes
    X_DG = 0 # Input feature dimension of graphs (no graph level input data)
    Y_DE = 3 # Output feature dimension of edges
    Y_DN = 4 # Output feature dimension of nodes
    Y_DG = 5 # Output feature dimension of graphs
    
    block = GNBlock(
        (X_DE,X_DN,X_DG) => (Y_DE,Y_DN,Y_DG)
    )

    @testset "example 1" begin
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

        @test size(y.ef) == (Y_DE, num_edges, batch_size)
        @test size(y.nf) == (Y_DN, num_nodes, batch_size)
        @test size(y.gf) == (Y_DG, batch_size)

        # Get the output graph edges of the 1st graph
        @test size(y.ef[:,:,1]) == (Y_DE, num_edges)

        # Get the output node edges of the 1st graph
        @test size(y.nf[:,:,1]) == (Y_DN, num_nodes)

        # Get the output graph edges of the 2nd graph
        @test size(y.gf[:,2]) == (Y_DG,)
    end

    @testset "example 2" begin
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
        @test size(efview(y_batched, :, :, 1)) == (Y_DE, num_edges_1) # edge features for graph 1
        @test size(nfview(y_batched, :, :, 1)) == (Y_DN, num_nodes_1)  # edge features for graph 1
        @test size(gfview(y_batched, :, 1)) == (Y_DG,) # graph features for graph 1
        @test size(efview(y_batched, :, :, 2)) == (Y_DE, num_edges_2) # edge features for graph 2
        @test size(nfview(y_batched, :, :, 2)) == (Y_DN, num_nodes_2) # node features for graph 2
        @test size(gfview(y_batched, :, 2)) == (Y_DG,) # graph features for graph 2

        # Copied array of features (less efficient) for a batch with different graph structures
        @test size(y.ef[1]) == (Y_DE, num_edges_1) # edge features for graph 1
        @test size(y.nf[1]) == (Y_DN, num_nodes_1)  # edge features for graph 1
        @test size(y.gf[1]) == (Y_DG,) # graph features for graph 1
        @test size(y.ef[2]) == (Y_DE, num_edges_2) # edge features for graph 2
        @test size(y.nf[2]) == (Y_DN, num_nodes_2) # node features for graph 2
        @test size(y.gf[2]) == (Y_DG,) # graph features for graph 2
    end

    @testset "example 3" begin
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

        @test size(y.ef) == (Y_DE, num_edges, batch_size)
        @test size(y.nf) == (Y_DN, num_nodes, batch_size)
        @test size(y.gf) == (Y_DG, batch_size)
    end
    
end

@testset "batch_inverse_2D" begin
    adj_mat_1 = [
        1 0 1;
        1 1 0;
        0 0 1;
    ]
    adj_mat_2 = [
        1 0 1 0;
        1 1 0 1;
        0 0 1 0;
        1 1 0 1
    ]
    adj_mats = [adj_mat_1, adj_mat_2]
    num_edges_1 = length(filter(isone, adj_mat_1))
    num_edges_2 = length(filter(isone, adj_mat_2))
    edge_dim = 10
    edge_features = [
        rand(Float32, edge_dim, num_edges_1),
        rand(Float32, edge_dim, num_edges_2),
    ]
    num_nodes_1 = size(adj_mat_1, 1)
    num_nodes_2 = size(adj_mat_2, 1)
    node_dim = 5
    node_features = [
        rand(Float32, node_dim, num_nodes_1),
        rand(Float32, node_dim, num_nodes_2),
    ]
    x = (
        graphs = adj_mats,
        ef = edge_features, 
        nf = node_features,
        gf = nothing # no graph level input features
    )
    x̂ = x |> batch |> unbatch
    @test x̂.graphs == x.graphs
    @test x̂.ef == x.ef
    @test x̂.nf == x.nf
    @test x̂.gf == x.gf
end

@testset "batch_inverse_3D" begin
    adj_mat = [
        1 0 1;
        1 1 0;
        0 0 1;
    ]
    num_edges = length(filter(isone, adj_mat))
    num_nodes = size(adj_mat, 1)
    batch_size = 2
    X_DE = 10
    X_DN = 5
    x = (
        graphs = adj_mat,
        ef = rand(Float32, X_DE, num_edges, batch_size), 
        nf = rand(Float32, X_DN, num_nodes, batch_size),
        gf = nothing # no graph level input features
    )
    x̂ = x |> batch |> unbatch
    @test x̂.graphs == x.graphs
    @test x̂.ef == x.ef
    @test x̂.nf == x.nf
    @test x̂.gf == x.gf
end

# @testset "padef" begin
#     adj_mat = [
#         1 0 1;
#         1 1 0;
#         0 0 1;
#     ]
#     ef_dim = 10
#     num_nodes = size(adj_mat, 1)
#     N2 = num_nodes^2
#     num_edges = length(filter(isone, adj_mat))
#     batch_size = 2
#     ef = rand(Float32, ef_dim, num_edges, batch_size)
#     padded = padef(adj_mat, ef)
#     @test size(padded) == (ef_dim, N2, batch_size)
#     adj_mat_1 = [
#         1 0 1;
#         1 1 0;
#         0 0 1;
#     ]
#     num_edges_1 = length(filter(isone, adj_mat_1))
#     adj_mat_2 = [
#         1 0 1 0;
#         1 1 0 1;
#         0 0 1 0;
#         1 1 0 1
#     ]
#     num_edges_2 = length(filter(isone, adj_mat_2))
#     adj_mats = [adj_mat_1, adj_mat_2]
#     max_num_nodes = maximum(size.(adj_mats, 1))
#     PN2 = max_num_nodes^2
#     batch_size = length(adj_mats)
#     efs = [
#         rand(Float32, ef_dim, num_edges_1),
#         rand(Float32, ef_dim, num_edges_2),
#     ]
#     padded = padef(adj_mats, efs)
#     @test size(padded) == (ef_dim, PN2, batch_size)
# end

# @testset "padnf" begin
#     adj_mat_1 = [
#         1 0 1;
#         1 1 0;
#         0 0 1;
#     ]
#     adj_mat_2 = [
#         1 0 1 0;
#         1 1 0 1;
#         0 0 1 0;
#         1 1 0 1
#     ]
#     adj_mats = [adj_mat_1, adj_mat_2]
#     batch_size = length(adj_mats)
#     max_num_nodes = maximum(first.(size.(adj_mats)))
#     nf_dim = 5
#     nfs = [
#         rand(Float32, nf_dim, size(adj_mat_1, 1)),
#         rand(Float32, nf_dim, size(adj_mat_2, 1)),
#     ]
#     padded = padnf(adj_mats, nfs)
#     @test size(padded) == (nf_dim, max_num_nodes, batch_size)
# end

# @testset "padadjmats" begin
#     PN = 4
#     adj_mats = [
#         rand(0:1, 3, 3), 
#         rand(0:1, PN, PN),
#     ]
#     B = length(adj_mats)
#     padded = padadjmats(adj_mats)
#     @test size(padded) == (PN,PN,B)
# end

# @testset "getnode2edgebroadcaster" begin
#     PN = 4
#     PN2 = PN^2
#     adj_mats = [
#         rand(0:1, 3, 3), 
#         rand(0:1, PN, PN),
#     ]
#     B = length(adj_mats)
#     padded = padadjmats(adj_mats)
#     broadcaster = getsrcnode2edgebroadcaster(padded)
#     @test size(broadcaster) == (PN,PN2,B)

#     PN = 3
#     PN2 = PN^2
#     adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
#     adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
#     adj_mats = [adj_mat1,adj_mat2]
#     B = length(adj_mats)
#     padded = padadjmats(adj_mats)
#     src_broadcaster = getsrcnode2edgebroadcaster(padded)
#     @test size(src_broadcaster) == (PN,PN2,B)
#     @test src_broadcaster[:,:,1] == [
#         1 0 0 0 0 0 1 0 0;
#         0 1 0 0 1 0 0 0 0;
#         0 0 0 0 0 0 0 0 1;
#     ]
#     @test src_broadcaster[:,:,2] == [
#         0 0 0 1 0 0 0 0 0;
#         0 0 0 0 0 0 0 1 0;
#         0 0 1 0 0 1 0 0 0;
#     ]
#     dst_broadcaster = getdstnode2edgebroadcaster(padded)
#     @test size(dst_broadcaster) == (PN,PN2,B)
#     @test dst_broadcaster[:,:,1] == [
#         1 1 0 0 0 0 0 0 0;
#         0 0 0 0 1 0 0 0 0;
#         0 0 0 0 0 0 1 0 1;
#     ]
#     @test dst_broadcaster[:,:,2] == [
#         0 0 1 0 0 0 0 0 0;
#         0 0 0 1 0 1 0 0 0;
#         0 0 0 0 0 0 0 1 0;
#     ]
# end

# @testset "GNGraphBatch" begin
#     adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
#     adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
#     adj_mats = [adj_mat1,adj_mat2]
#     batch = GNGraphBatch(adj_mats)
#     @test !isnothing(batch)
# end

# @testset "getedgefninput (edge, node, and graph features)" begin
#     adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
#     adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
#     adj_mats = [adj_mat1,adj_mat2]
#     graphs = GNGraphBatch(adj_mats)
#     PN = first(size(graphs.padded_adj_mats))
#     PN2 = PN^2
#     DIM_E = 10
#     DIM_N = 5
#     DIM_G = 3
#     B = 2
#     edge_features = rand(Float32, DIM_E, PN2, B)
#     node_features = rand(Float32, DIM_N, PN, B)
#     graph_features = rand(Float32, DIM_G, 1, B)
#     edge_fn_input = getedgefninput(graphs, edge_features, node_features, graph_features)
#     @test size(edge_fn_input) == (DIM_E+2DIM_N+DIM_G, PN2, B)
# end

# @testset "getedgefninput (graph features only)" begin
#     adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
#     adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
#     adj_mats = [adj_mat1,adj_mat2]
#     graphs = GNGraphBatch(adj_mats)
#     PN = first(size(graphs.padded_adj_mats))
#     PN2 = PN^2
#     DIM_G = 3
#     B = 2
#     graph_features = rand(Float32, DIM_G, 1, B)
#     edge_fn_input = getedgefninput(graphs, nothing, nothing, graph_features)
#     @test size(edge_fn_input) == (DIM_G, PN2, B)
# end

# @testset "getedgefninput (node features only)" begin
#     adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
#     adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
#     adj_mats = [adj_mat1,adj_mat2]
#     graphs = GNGraphBatch(adj_mats)
#     PN = first(size(graphs.padded_adj_mats))
#     PN2 = PN^2
#     DIM_N = 5
#     B = 2
#     node_features = rand(Float32, DIM_N, PN, B)
#     edge_fn_input = getedgefninput(graphs, nothing, node_features, nothing)
#     @test size(edge_fn_input) == (2DIM_N, PN2, B)
# end

# @testset "getedgefninput (node and graph features)" begin
#     adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
#     adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
#     adj_mats = [adj_mat1,adj_mat2]
#     graphs = GNGraphBatch(adj_mats)
#     PN = first(size(graphs.padded_adj_mats))
#     PN2 = PN^2
#     DIM_N = 5
#     DIM_G = 3
#     B = 2
#     node_features = rand(Float32, DIM_N, PN, B)
#     graph_features = rand(Float32, DIM_G, 1, B)
#     edge_fn_input = getedgefninput(graphs, nothing, node_features, graph_features)
#     @test size(edge_fn_input) == (2DIM_N+DIM_G, PN2, B)
# end

# @testset "getedgefninput (edge features only)" begin
#     adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
#     adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
#     adj_mats = [adj_mat1,adj_mat2]
#     graphs = GNGraphBatch(adj_mats)
#     PN = first(size(graphs.padded_adj_mats))
#     PN2 = PN^2
#     DIM_E = 10
#     B = 2
#     edge_features = rand(Float32, DIM_E, PN2, B)
#     edge_fn_input = getedgefninput(graphs, edge_features, nothing, nothing)
#     @test size(edge_fn_input) == (DIM_E, PN2, B)
# end

# @testset "getedgefninput (edge and graph features)" begin
#     adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
#     adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
#     adj_mats = [adj_mat1,adj_mat2]
#     graphs = GNGraphBatch(adj_mats)
#     PN = first(size(graphs.padded_adj_mats))
#     PN2 = PN^2
#     DIM_E = 10
#     DIM_G = 3
#     B = 2
#     edge_features = rand(Float32, DIM_E, PN2, B)
#     graph_features = rand(Float32, DIM_G, 1, B)
#     edge_fn_input = getedgefninput(graphs, edge_features, nothing, graph_features)
#     @test size(edge_fn_input) == (DIM_E+DIM_G, PN2, B)
# end

# @testset "getedgefninput (edge features and node features)" begin
#     adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
#     adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
#     adj_mats = [adj_mat1,adj_mat2]
#     graphs = GNGraphBatch(adj_mats)
#     PN = first(size(graphs.padded_adj_mats))
#     PN2 = PN^2
#     DIM_E = 10
#     DIM_N = 5
#     B = 2
#     edge_features = rand(Float32, DIM_E, PN2, B)
#     node_features = rand(Float32, DIM_N, PN, B)
#     edge_fn_input = getedgefninput(graphs, edge_features, node_features, nothing)
#     @test size(edge_fn_input) == (DIM_E+2DIM_N, PN2, B)
# end

@testset "GNBlock" begin
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
    @test size(y.ef) == (Y_DE, num_edges, batch_size)
    @test size(y.nf) == (Y_DN, num_nodes, batch_size)
    @test size(y.gf) == (Y_DG, batch_size)
end

# @testset "edge2nodebroadcaster" begin
#     adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
#     adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
#     adj_mats = [adj_mat1,adj_mat2]
#     graphs = GNGraphBatch(adj_mats)
#     expected_1 = [
#         1.0  0.0  0.0;
#         1.0  0.0  0.0;
#         0.0  0.0  0.0;
#         0.0  0.0  0.0;
#         0.0  1.0  0.0;
#         0.0  0.0  0.0;
#         0.0  0.0  1.0;
#         0.0  0.0  0.0;
#         0.0  0.0  1.0;
#     ]
#     expected_2 = [
#         0.0  0.0  0.0;
#         0.0  0.0  0.0;
#         1.0  0.0  0.0;
#         0.0  1.0  0.0;
#         0.0  0.0  0.0;
#         0.0  1.0  0.0;
#         0.0  0.0  0.0;
#         0.0  0.0  1.0;
#         0.0  0.0  0.0;
#     ]
#     @test graphs.edge2node_broadcaster[:,:,1] == expected_1
#     @test graphs.edge2node_broadcaster[:,:,2] == expected_2
# end

@testset "GNCore" begin
    dims = (DE, DN, DG) = 3, 4, 5
    core = GNCore(dims)
    adj_mat = adj_mat = [
        1 0 1;
        1 1 0;
        0 0 1;
    ]
    num_nodes = size(adj_mat, 1)
    num_edges = length(filter(isone, adj_mat))
    batch_size = 2
    edge_features = rand(Float32, DE, num_edges, batch_size)
    node_features = rand(Float32, DN, num_nodes, batch_size)
    graph_features = rand(Float32, DG, batch_size)
    x = (
        graphs=adj_mat, # All graphs in this batch have same structure
        ef=edge_features, # (DE, num_edges, batch_size)
        nf=node_features, # (DN, num_nodes, batch_size)
        gf=graph_features # no input graph features
    ) |> batch
    y = core(x) |> unbatch
    @test size(y.ef) == (DE, num_edges, batch_size)
    @test size(y.nf) == (DN, num_nodes, batch_size)
    @test size(y.gf) == (DG, batch_size)
end

@testset "GNCoreList" begin
    dims = (DE, DN, DG) = 3, 4, 5
    core_list = GNCoreList([GNCore(dims), GNCore(dims)])
    adj_mat = adj_mat = [
        1 0 1;
        1 1 0;
        0 0 1;
    ]
    num_nodes = size(adj_mat, 1)
    num_edges = length(filter(isone, adj_mat))
    batch_size = 2
    edge_features = rand(Float32, DE, num_edges, batch_size)
    node_features = rand(Float32, DN, num_nodes, batch_size)
    graph_features = rand(Float32, DG, batch_size)
    x = (
        graphs=adj_mat, # All graphs in this batch have same structure
        ef=edge_features, # (DE, num_edges, batch_size)
        nf=node_features, # (DN, num_nodes, batch_size)
        gf=graph_features # no input graph features
    ) |> batch
    y = core_list(x) |> unbatch
    @test size(y.ef) == (DE, num_edges, batch_size)
    @test size(y.nf) == (DN, num_nodes, batch_size)
    @test size(y.gf) == (DG, batch_size)
end