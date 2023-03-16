function padcat(v::AbstractVector)
    @assert (isone∘length∘unique)(size.(v, 1))
    D = size(first(v), 1)
    B = length(v)
    padded_block_size = maximum(size.(v, 2))
    m = zeros(Float32, D, padded_block_size, B)
    for (graph_idx,graph) in enumerate(v)
        block_size = size(graph, 2)
        m[1:D, 1:block_size, graph_idx]
    end
    m
end

function padmat(m::AbstractMatrix, pd1, pd2)
    @assert pd1 >= size(m,1)
    @assert pd2 >= size(m,2)
    d1, d2 = size(m)
    padded = zeros(Float32, pd1, pd2)
    padded[1:d1, 1:d2] .= m
    padded
end

function paddedbatch(v)
    @assert (isone∘length∘unique)(size.(v, 1))
    d1 = size(v[1], 1)
    d2 = maximum(size.(v, 2))
    padded = padmat.(v, d1, d2)
    reduce((a,b)->cat(a,b; dims=3), padded)
end