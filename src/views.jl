"""
    efview(t::NamedTuple, d1, d2, d3)

Returns an array view of the edge features contained in the batched output.
"""
function efview(t::NamedTuple, d1, d2, d3)
    @assert issubset(Set((:graphs, :ef)), Set(keys(t)))
    (; graphs, ef) = t
    efview(graphs, ef, d1, d2, d3)
end

efview(::GNGraphBatch, ::Nothing, d1, d2, d3) = nothing

function efview(graphs::GNGraphBatch, ef, d1, d2, d3)
    if length(graphs.adj_mats) == 1
        return efview(graphs.adj_mats[1], ef, d1, d2, d3)
    end
    efview(graphs.adj_mats, ef, d1, d2, d3)
end

function efview(adj_mat::AbstractMatrix, ef, d1, d2, d3)
    e_idx = findall(isone, view(adj_mat, :))[d2]
    @view ef[d1, e_idx, d3]
end

function efview(adj_mats::AbstractVector, ef, d1, d2, d3::Integer)
    # TODO: some inefficiency here, as padded_adj_mats is already computed
    padded_adj_mat = @view padadjmats(adj_mats)[:, :, d3]
    unbatched_view = @view ef[:, findall(isone, view(padded_adj_mat, :)), d3]
    @view unbatched_view[d1, d2]
end

"""
    nfview(t::NamedTuple, d1, d2, d3)

Returns an array view of the node features contained in the batched output.
"""
function nfview(t::NamedTuple, d1, d2, d3)
    @assert issubset(Set((:graphs, :nf)), Set(keys(t)))
    (; graphs, nf) = t
    nfview(graphs, nf, d1, d2, d3)
end

nfview(::GNGraphBatch, ::Nothing, d1, d2, d3) = nothing

function nfview(graphs::GNGraphBatch, nf, d1, d2, d3)
    if length(graphs.adj_mats) == 1
        return nfview(graphs.adj_mats[1], nf, d1, d2, d3)
    end
    nfview(graphs.adj_mats, nf, d1, d2, d3)
end

function nfview(adj_mat::AbstractMatrix, nf, d1, d2, d3)
    @view nf[d1, d2, d3]
end

function nfview(adj_mats::AbstractVector, nf, d1, d2, d3::Integer)
    adj_mat = adj_mats[d3]
    unbatched_view = @view nf[:, 1:size(adj_mat,1), d3]
    @view unbatched_view[d1, d2]
end

"""
    gfview(t::NamedTuple, d1, d2)

Returns an array view of the graph features contained in the batched output.
"""
function gfview(t::NamedTuple, d1, d2)
    @assert issubset(Set((:graphs, :gf)), Set(keys(t)))
    (; graphs, gf) = t
    gfview(graphs, gf, d1, d2)
end

gfview(::GNGraphBatch, ::Nothing, d1, d2) = nothing

function gfview(graphs::GNGraphBatch, gf, d1, d2)
    @view gf[d1, 1, d2]
end
