using GraphNets
using Test

@testset "Test padadjmats" begin
    PN = 4
    adj_mats = [
        rand(0:1, 3, 3), 
        rand(0:1, PN, PN),
    ]
    B = length(adj_mats)
    padded = padadjmats(adj_mats)
    @test size(padded) == (PN,PN,B)
end

@testset "Test getsrcnodebroadcaster" begin
    PN = 4
    PN2 = PN^2
    adj_mats = [
        rand(0:1, 3, 3), 
        rand(0:1, PN, PN),
    ]
    B = length(adj_mats)
    padded = padadjmats(adj_mats)
    broadcaster = getsrcnodebroadcaster(padded)
    @test size(broadcaster) == (PN,PN2,B)

    PN = 3
    PN2 = PN^2
    adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
    adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
    adj_mats = [adj_mat1,adj_mat2]
    B = length(adj_mats)
    padded = padadjmats(adj_mats)
    src_broadcaster = getsrcnodebroadcaster(padded)
    @test size(src_broadcaster) == (PN,PN2,B)
    @test src_broadcaster[:,:,1] == [
        1 0 0 0 0 0 1 0 0;
        0 1 0 0 1 0 0 0 0;
        0 0 0 0 0 0 0 0 1;
    ]
    @test src_broadcaster[:,:,2] == [
        0 0 0 1 0 0 0 0 0;
        0 0 0 0 0 0 0 1 0;
        0 0 1 0 0 1 0 0 0;
    ]
    dst_broadcaster = getdstnodebroadcaster(padded)
    @test size(dst_broadcaster) == (PN,PN2,B)
    @test dst_broadcaster[:,:,1] == [
        1 1 0 0 0 0 0 0 0;
        0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 0 1 0 1;
    ]
    @test dst_broadcaster[:,:,2] == [
        0 0 1 0 0 0 0 0 0;
        0 0 0 1 0 1 0 0 0;
        0 0 0 0 0 0 0 1 0;
    ]
end

@testset "Test GNGraphBatch" begin
    adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
    adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
    adj_mats = [adj_mat1,adj_mat2]
    batch = GNGraphBatch(adj_mats)
    @test !isnothing(batch)
end

@testset "Test getedgefninput (edge, node, and graph features)" begin
    adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
    adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
    adj_mats = [adj_mat1,adj_mat2]
    graphs = GNGraphBatch(adj_mats)
    PN = first(size(graphs.padded_adj_mats))
    PN2 = PN^2
    DIM_E = 10
    DIM_N = 5
    DIM_G = 3
    B = 2
    edge_features = rand(Float32, DIM_E, PN2, B)
    node_features = rand(Float32, DIM_N, PN, B)
    graph_features = rand(Float32, DIM_G, 1, B)
    edge_fn_input = getedgefninput(graphs, edge_features, node_features, graph_features)
    @test size(edge_fn_input) == (DIM_E+2DIM_N+DIM_G, PN2, B)
end

@testset "Test getedgefninput (graph features only)" begin
    adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
    adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
    adj_mats = [adj_mat1,adj_mat2]
    graphs = GNGraphBatch(adj_mats)
    PN = first(size(graphs.padded_adj_mats))
    PN2 = PN^2
    DIM_G = 3
    B = 2
    graph_features = rand(Float32, DIM_G, 1, B)
    edge_fn_input = getedgefninput(graphs, nothing, nothing, graph_features)
    @test size(edge_fn_input) == (DIM_G, PN2, B)
end

@testset "Test getedgefninput (node features only)" begin
    adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
    adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
    adj_mats = [adj_mat1,adj_mat2]
    graphs = GNGraphBatch(adj_mats)
    PN = first(size(graphs.padded_adj_mats))
    PN2 = PN^2
    DIM_N = 5
    B = 2
    node_features = rand(Float32, DIM_N, PN, B)
    edge_fn_input = getedgefninput(graphs, nothing, node_features, nothing)
    @test size(edge_fn_input) == (2DIM_N, PN2, B)
end

@testset "Test getedgefninput (node and graph features)" begin
    adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
    adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
    adj_mats = [adj_mat1,adj_mat2]
    graphs = GNGraphBatch(adj_mats)
    PN = first(size(graphs.padded_adj_mats))
    PN2 = PN^2
    DIM_N = 5
    DIM_G = 3
    B = 2
    node_features = rand(Float32, DIM_N, PN, B)
    graph_features = rand(Float32, DIM_G, 1, B)
    edge_fn_input = getedgefninput(graphs, nothing, node_features, graph_features)
    @test size(edge_fn_input) == (2DIM_N+DIM_G, PN2, B)
end

@testset "Test getedgefninput (edge features only)" begin
    adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
    adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
    adj_mats = [adj_mat1,adj_mat2]
    graphs = GNGraphBatch(adj_mats)
    PN = first(size(graphs.padded_adj_mats))
    PN2 = PN^2
    DIM_E = 10
    B = 2
    edge_features = rand(Float32, DIM_E, PN2, B)
    edge_fn_input = getedgefninput(graphs, edge_features, nothing, nothing)
    @test size(edge_fn_input) == (DIM_E, PN2, B)
end

@testset "Test getedgefninput (edge and graph features)" begin
    adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
    adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
    adj_mats = [adj_mat1,adj_mat2]
    graphs = GNGraphBatch(adj_mats)
    PN = first(size(graphs.padded_adj_mats))
    PN2 = PN^2
    DIM_E = 10
    DIM_G = 3
    B = 2
    edge_features = rand(Float32, DIM_E, PN2, B)
    graph_features = rand(Float32, DIM_G, 1, B)
    edge_fn_input = getedgefninput(graphs, edge_features, nothing, graph_features)
    @test size(edge_fn_input) == (DIM_E+DIM_G, PN2, B)
end

@testset "Test getedgefninput (edge features and node features)" begin
    adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
    adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
    adj_mats = [adj_mat1,adj_mat2]
    graphs = GNGraphBatch(adj_mats)
    PN = first(size(graphs.padded_adj_mats))
    PN2 = PN^2
    DIM_E = 10
    DIM_N = 5
    B = 2
    edge_features = rand(Float32, DIM_E, PN2, B)
    node_features = rand(Float32, DIM_N, PN, B)
    edge_fn_input = getedgefninput(graphs, edge_features, node_features, nothing)
    @test size(edge_fn_input) == (DIM_E+2DIM_N, PN2, B)
end

@testset "GNBlock" begin
    from = X_DE,X_DN,X_DG = 1,0,0
    to = Y_DE,Y_DN,Y_DG = 2,3,4
    N,B = 3,2
    N2 = N^2
    adj_mats = [rand(0:1, N, N) for _ in 1:B]
    x = (
        graphs=GNGraphBatch(adj_mats),
        edge_features=Float32.(rand(X_DE,N2,B)), 
        node_features=nothing, 
        graph_features=nothing
    )
    block = GNBlock(from=>to)
    @test size(block(x)) == (Y_DE,N2,B)
end