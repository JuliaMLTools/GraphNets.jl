module GraphNets

export func

"""
    func(x)

Returns double the number a `x` plus `1`.
"""
func(x) = 2x + 1


include("imports.jl")
include("util.jl")

include("gngraphbatch.jl")
export GNGraphBatch, padadjmats, getsrcnodebroadcaster, getdstnodebroadcaster

include("edgefninput.jl")
export getedgefninput

include("gnblock.jl")
export GNBlock

import SnoopPrecompile
include("other/precompile.jl")

end