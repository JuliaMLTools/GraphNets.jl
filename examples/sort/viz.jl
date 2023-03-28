function getinputgraph(x)
    nodes = Flux.onecold(x.nf)
    n = length(nodes)
    g = EuclidGraph(
        ngon(length(nodes)),
        fully_connected=true, 
        node_style=(node) -> NodeStyle(
            value=(node) -> node.features[node.idx],
        ),
    )
    g(nodes)
end

function gettargetgraph(y)
    nodes = Flux.onecold(y.nf, 0:1)
    edges = Flux.onecold(y.ef, 0:1)
    n = length(nodes)
    g = EuclidGraph(
        ngon(n),
        adj_mat=reshape(edges, n, n),
        node_style=(node) -> NodeStyle(
            stroke="#ccc",
            inner_fill=(isone(node.features[node.idx]) ? "green" : "#fff"),
            value=(node) -> nothing
        ),
        edge_style=(edge) -> EdgeStyle(
            stroke="green",
        )
    )
    g(nodes, edges)
end