function unbatch(t::NamedTuple)
    @assert Set(keys(t)) == Set((:graphs, :ef, :nf, :gf))
    (; graphs, ef, nf, gf) = t
    @assert !isnothing(ef) || !isnothing(nf) || !isnothing(gf)
    unbatch(graphs,ef,nf,gf)
end

function unbatch(graphs::GNGraphBatch, ef, nf, gf)
    @assert !isnothing(ef) || !isnothing(nf) || !isnothing(gf)
    if length(graphs.adj_mats) == 1
        return unbatch(graphs.adj_mats[1], ef, nf, gf)
    end
    unbatch(graphs.adj_mats, ef, nf, gf)
end

function unbatch(adj_mat::AbstractMatrix, ef, nf, gf)
    @assert !isnothing(ef) || !isnothing(nf) || !isnothing(gf)
    (
        graphs=adj_mat,
        ef=unbatchef(adj_mat, ef),
        nf=nf,
        gf=unbatchgf(adj_mat, gf),
    )
end

function unbatch(adj_mats::AbstractVector, ef, nf, gf)
    @assert !isempty(ef) || !isempty(nf) || !isempty(gf)
    (
        graphs=adj_mats,
        ef=unbatchef(adj_mats, ef),
        nf=unbatchnf(adj_mats, nf),
        gf=unbatchgf(adj_mats, gf),
    )
end

unbatchef(adj_mats, ef) = unpadef(adj_mats, ef)
unbatchef(adj_mats, ::Nothing) = nothing

unbatchnf(adj_mats, nf) = unpadnf(adj_mats, nf)
unbatchnf(adj_mats, ::Nothing) = nothing

unbatchgf(adj_mats, gf) = unpadgf(adj_mats, gf)
unbatchgf(adj_mats, ::Nothing) = nothing