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
export GNGraphBatch, padadjmats, getsrcnode2edgebroadcaster, getdstnode2edgebroadcaster

include("edgefninput.jl")
export getedgefninput

include("nodefninput.jl")
export getnodefninput

include("graphfninput.jl")
export getgraphfninput

include("gnblock.jl")
export GNBlock

import SnoopPrecompile
include("other/precompile.jl")

end