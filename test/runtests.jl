using GraphNets
using Test

@testset "padadjmats" begin
    PN = 4
    adj_mats = [
        rand(0:1, 3, 3), 
        rand(0:1, PN, PN),
    ]
    B = length(adj_mats)
    padded = padadjmats(adj_mats)
    @test size(padded) == (PN,PN,B)
end

@testset "getnode2edgebroadcaster" begin
    PN = 4
    PN2 = PN^2
    adj_mats = [
        rand(0:1, 3, 3), 
        rand(0:1, PN, PN),
    ]
    B = length(adj_mats)
    padded = padadjmats(adj_mats)
    broadcaster = getsrcnode2edgebroadcaster(padded)
    @test size(broadcaster) == (PN,PN2,B)

    PN = 3
    PN2 = PN^2
    adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
    adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
    adj_mats = [adj_mat1,adj_mat2]
    B = length(adj_mats)
    padded = padadjmats(adj_mats)
    src_broadcaster = getsrcnode2edgebroadcaster(padded)
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
    dst_broadcaster = getdstnode2edgebroadcaster(padded)
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

@testset "GNGraphBatch" begin
    adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
    adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
    adj_mats = [adj_mat1,adj_mat2]
    batch = GNGraphBatch(adj_mats)
    @test !isnothing(batch)
end

@testset "getedgefninput (edge, node, and graph features)" begin
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

@testset "getedgefninput (graph features only)" begin
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

@testset "getedgefninput (node features only)" begin
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

@testset "getedgefninput (node and graph features)" begin
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

@testset "getedgefninput (edge features only)" begin
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

@testset "getedgefninput (edge and graph features)" begin
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

@testset "getedgefninput (edge features and node features)" begin
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
        graphs = GNGraphBatch(adj_mats),
        ef = Float32.(rand(X_DE,N2,B)),
        nf = nothing, 
        gf = nothing,
    )
    block = GNBlock(from=>to)
    _, y_e, y_n, y_g = block(x)
    @test size(y_e) == (Y_DE,N2,B)
    @test size(y_n) == (Y_DN,N,B)
    @test size(y_g) == (Y_DG,1,B)
end

@testset "edge2nodebroadcaster" begin
    adj_mat1 = Float32.([1 0 1; 1 1 0; 0 0 1])
    adj_mat2 = Float32.([0 1 0; 0 0 1; 1 1 0])
    adj_mats = [adj_mat1,adj_mat2]
    graphs = GNGraphBatch(adj_mats)
    expected_1 = [
        1.0  0.0  0.0;
        1.0  0.0  0.0;
        0.0  0.0  0.0;
        0.0  0.0  0.0;
        0.0  1.0  0.0;
        0.0  0.0  0.0;
        0.0  0.0  1.0;
        0.0  0.0  0.0;
        0.0  0.0  1.0;
    ]
    expected_2 = [
        0.0  0.0  0.0;
        0.0  0.0  0.0;
        1.0  0.0  0.0;
        0.0  1.0  0.0;
        0.0  0.0  0.0;
        0.0  1.0  0.0;
        0.0  0.0  0.0;
        0.0  0.0  1.0;
        0.0  0.0  0.0;
    ]
    @test graphs.edge2node_broadcaster[:,:,1] == expected_1
    @test graphs.edge2node_broadcaster[:,:,2] == expected_2
end

@testset "GNCore" begin
    dims = DE, DN, DG = (10,5,3)
    N,B = 3,2
    N2 = N^2
    adj_mats = [rand(0:1, N, N) for _ in 1:B]
    x = (
        graphs = GNGraphBatch(adj_mats),
        ef = rand(Float32, DE, N2, B),
        nf = rand(Float32, DN, N, B), 
        gf = rand(Float32, DG, 1, B)
    )
    core = GNCore(dims)
    _, y_e, y_n, y_g = core(x)
    @test size(y_e) == (DE,N2,B)
    @test size(y_n) == (DN,N,B)
    @test size(y_g) == (DG,1,B)
end

@testset "GNCoreList" begin
    dims = CE, CN, CG = (10,5,3) # Channel dims (edge, node, graph) for each core
    N, B = 3, 2 # N = Node count, B = Batch size
    PE = N^2 # PE = Padded Edge Count
    adj_mats = [rand(0:1, N, N) for _ in 1:B] # Randomize adjacency matrices
    x = (
        graphs = GNGraphBatch(adj_mats),
        ef = rand(Float32, CE, PE, B),
        nf = rand(Float32, CN, N, B), 
        gf = rand(Float32, CG, 1, B)
    )
    core_list = GNCoreList([GNCore(dims), GNCore(dims)])
    y = core_list(x)
    @test size(y.ef) == (CE, PE, B)
    @test size(y.nf) == (CN, N, B)
    @test size(y.gf) == (CG, 1, B)
end