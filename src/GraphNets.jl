module GraphNets

export func

"""
    func(x)

Returns double the number a `x` plus `1`.
"""
func(x) = 2x + 1


include("imports.jl")

include("util.jl")
export padcat, padmat, paddedbatch, getedgefeatures, getnodefeatures, getgraphfeatures

include("checks.jl")

include("gngraphbatch.jl")
export GNGraphBatch, padadjmats, getsrcnode2edgebroadcaster, getdstnode2edgebroadcaster

include("batch.jl")
export batch

include("unbatch.jl")
export unbatch

include("pad.jl")
export padef, padnf

include("unpad.jl")

include("edgefninput.jl")
export getedgefninput

include("nodefninput.jl")
export getnodefninput

include("graphfninput.jl")
export getgraphfninput

include("gnfeedforward.jl")
export GNFeedForward

include("gnblock.jl")
export GNBlock

include("gncore.jl")
export GNCore

include("gncorelist.jl")
export GNCoreList

include("gngraphnorm.jl")
export GNGraphNorm

include("views.jl")
export efview, nfview, gfview

import SnoopPrecompile
include("other/precompile.jl")

end