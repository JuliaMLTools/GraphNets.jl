cd(@__DIR__)
using GraphNets

#################### SETUP ####################
X_DE = 10 # Input feature dimension of edges
X_DN = 0 # No node input data (dimension zero)
X_DG = 0 # No graph input data (dimension zero)
Y_DE = 1 # Output feature dimension of edges
Y_DN = 2 # Output feature dimension of nodes
Y_DG = 3 # Output feature dimension of graphs
block = GNBlock(
    (X_DE,X_DN,X_DG) => (Y_DE,Y_DN,Y_DG)
)
################################################


####
# Example #1: GNN input graphs have same structure (same graph), but different features
####

adj_mat = [
    1 0 1;
    1 1 0;
    0 0 1;
] # Adjacency matrix

adj_mats = [adj_mat, adj_mat]
G = length(adj_mats) # Number of graphs
N = size(adj_mat, 1) # Number of nodes in each graph
graphs = GNGraphBatch(adj_mats)

# Edge features for graph 1
edge_features_1 = rand(Float32, X_DE, N^2) # (X_DE, N^2)

# Edge features for graph 2
edge_features_2 = rand(Float32, X_DE, N^2) # (X_DE, N^2)

# Batch of edge features
edge_features = cat(edge_features_1, edge_features_2; dims=3) # (X_DE, N^2, G) 

x = (
    graphs=graphs,
    edge_features=edge_features,
    node_features=nothing,
    graph_features=nothing,
)

_, y_e, y_n, y_g = block(x)
@assert size(y_e) == (Y_DE, N^2, G)
@assert size(y_n) == (Y_DN, N, G)
@assert size(y_g) == (Y_DG, 1, G)




# Example #2: GNN input graphs have different structures (different graphs).

adj_mat_1 = [
    1 0 1;
    1 1 0;
    0 0 1;
] # Adjacency matrix for graph 1

adj_mat_2 = [
    0 1 0 1;
    0 0 1 0;
    1 1 0 1;
    0 0 1 0;
] # Adjacency matrix for graph 2

adj_mats = [adj_mat_1, adj_mat_2]
G = length(adj_mats) # Number of graphs
graphs = GNGraphBatch(adj_mats)

N1 = size(adj_mat_1, 1) # Number of nodes in graph 1
N2 = size(adj_mat_2, 1) # Number of nodes in graph 2

edge_features = paddedbatch(
    [
        # Edge features for graph 1
        rand(Float32, X_DE, N1^2), # (X_DE, N1^2)
        
        # Edge features for graph 2
        rand(Float32, X_DE, N2^2), # (X_DE, N2^2)
    ]
)

# EBS (edge block size) is the maximum number of edges of any graph in the batch
EBS = size(edge_features, 2)
# NBS (node block size) is the maximum number of nodes of any graph in the batch
NBS = maximum([N1,N2])

x = (
    graphs=graphs,
    edge_features=edge_features,
    node_features=nothing,
    graph_features=nothing,
)

_, y_e, y_n, y_g = block(x)
@assert size(y_e) == (Y_DE, EBS, G)
@assert size(y_n) == (Y_DN, NBS, G)
@assert size(y_g) == (Y_DG, 1, G)