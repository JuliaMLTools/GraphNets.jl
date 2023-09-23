using GraphNets

in_dims = (X_DE, X_DN, X_DG) = (0, 2, 0)
out_dims = (Y_DE, Y_DN, Y_DG) = (2, 2, 2)
encoder = GNBlock(in_dims => out_dims)
decoder = GNBlock(out_dims => out_dims)

function block(x)
    x2 = encoder(x)
    decoder(x2)
end

function layer(m, x)
    (; graphs, ef, nf, gf) = x
    edge_fn_input = getedgefninput(graphs, ef, nf, gf)
    
    # display(edge_fn_input)
    
    h_ef = m.edgefn(edge_fn_input)
    node_fn_input = getnodefninput(graphs, h_ef, nf, gf)
    
    # println("node_fn_input ***[")
    # display(node_fn_input)
    # println("***]")
    # println()

    h_nf = m.nodefn(node_fn_input)

    # @show isnothing.([h_ef, h_nf, gf])

    graph_fn_input = getgraphfninput(graphs, h_ef, h_nf, gf)
    
    # println("graph_fn_input ***[")
    # display(graph_fn_input)
    # println("***]")
    # println()
    
    h_gf = m.graphfn(graph_fn_input)
    (graphs=graphs, ef=h_ef, nf=h_nf, gf=h_gf) |> zerodim2nothing
end



println("1Batch A -----------------")
foo = layer(encoder, x_1batch)
println("\n1Batch B -----------------")
car = layer(encoder, x_1batch_alt)
println("\nN-Batch -----------------")
bar = layer(encoder, x_nbatch)

display(foo.nf)
display(car.nf)
println()
display(bar.nf)

display(foo.ef)
display(car.ef)
println()
display(bar.ef)


# ******* BUG IS HERE ********
display(foo.gf)
display(car.gf)
println()
display(bar.gf)
#*****************************

foo2 = layer(decoder, foo)
car2 = layer(decoder, car)

# ******* check here next for bugs ********
bar2 = layer(decoder, bar)
#*****************************

display(foo2.ef)
display(car2.ef)
println()
display(bar2.ef)






adj_mat_A = [
    1 1
    1 1
]
adj_mat_B = [
    1 1 1
    1 1 1
    1 1 1
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
x_1batch_alt = (
    graphs=[adj_mat_B], # All graphs in this batch have same structure
    ef=nothing,
    nf=node_features[2:2],
    gf=nothing # no input graph features
) |> batch
ŷ_1batch = block(x_1batch)
nf_1batch_1 = nfview(ŷ_1batch, :, :, 1)
ef_1batch_1 = efview(ŷ_1batch, :, :, 1)

x_nbatch = (
    graphs=[adj_mat_A, adj_mat_B], # All graphs in this batch have same structure
    ef=nothing,
    nf=node_features[:],
    gf=nothing # no input graph features
) |> batch
ŷ_nbatch = block(x_nbatch)
nf_nbatch_1 = nfview(ŷ_nbatch, :, :, 1)
ef_nbatch_1 = efview(ŷ_nbatch, :, :, 1)

size(nf_1batch_1) == size(nf_nbatch_1)
size(ef_1batch_1) == size(ef_nbatch_1)
all(nf_1batch_1 .≈ nf_nbatch_1)


