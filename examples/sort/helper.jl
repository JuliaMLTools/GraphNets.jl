function getedgetargets(nodes)
    n = length(nodes)
    nodes_idx = collect(zip(1:n, nodes))
    sorted = first.(sort(nodes_idx; lt=(a,b)->last(a) < last(b)))
    enabled_edges = collect(zip(sorted[1:end-1], sorted[2:end]))
    edge_targets_mat = zeros(Int, n, n)
    for (i,j) in enabled_edges
        edge_targets_mat[i,j] = 1
    end
    edge_targets = edge_targets_mat[:]
    edge_targets
end