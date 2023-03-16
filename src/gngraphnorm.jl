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
    (
        graphs = x.graphs,
        ef = m.edgeln(x.ef),
        nf = m.nodeln(x.nf),
        gf = m.graphln(x.gf),
    )
end