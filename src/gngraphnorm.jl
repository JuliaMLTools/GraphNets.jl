struct GNGraphNorm
    edgeln
    nodeln
    graphln
end

Functors.@functor GNGraphNorm

"""
"""
function GNGraphNorm(dims; dropout=0)
    @assert all(dims .> (0,0,0))
    de, dn, dg = dims
    GNGraphNorm(
        LayerNorm(de),
        LayerNorm(dn),
        LayerNorm(dg),
    )
end

function (m::GNGraphNorm)(x)
    graphs, ef, nf, gf = x
    (
        graphs,
        m.edgeln(ef),
        m.nodeln(nf),
        m.graphln(gf),
    )
end