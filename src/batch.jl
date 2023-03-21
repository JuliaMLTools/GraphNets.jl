function batch(t::NamedTuple)
    @assert Set(keys(t)) == Set((:graphs, :ef, :nf, :gf))
    (; graphs, ef, nf, gf) = t
    @assert !isnothing(ef) || !isnothing(nf) || !isnothing(gf)
    checks(graphs,ef,nf,gf)
    (
        graphs = batchgraphs(graphs),
        ef = batchef(graphs, ef),
        nf = batchnf(graphs, nf), 
        gf = batchgf(graphs, gf),
    )
end

batchgraphs(adj_mat::AbstractMatrix) = GNGraphBatch([adj_mat])
batchgraphs(adj_mats::AbstractVector) = GNGraphBatch(adj_mats)

batchef(adj_mats, ef) = padef(adj_mats, ef)
batchef(adj_mats, ::Nothing) = nothing

batchnf(adj_mats, nf) = padnf(adj_mats, nf)
batchnf(adj_mats, ::Nothing) = nothing

batchgf(adj_mats, gf) = padgf(adj_mats, gf)
batchgf(adj_mats, ::Nothing) = nothing