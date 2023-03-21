struct GNFeedForward{E,N,G}
    eff::E
    nff::N
    gff::G
end

Functors.@functor GNFeedForward

"""
GNFeedForward(input_dim::Integer; dropout=0)

Initializes an instance of the **`GNFeedForward`** type, representing a simple linear layer followed by a non-linearity.

The following keyword arguments are supported:
- `dropout` (Defaults to 0)

A **`GNFeedForward`** instance accepts an input array **`x`** of dimensions (C, T, B) and outputs an array of dimensions (C, T, B). "C" is the channel size (embedding dimension). "T" is the block size (number of input tokens). "B" is the batch size.

## Examples:

```julia
C,T,B = 8,3,4
ff = GNFeedForward(C)
@assert size(ff(rand(Float32, C, T, B))) == (C, T, B)
```
"""
function GNFeedForward(dims; dropout=0)
    @assert all(dims .> (0,0,0))
    de, dn, dg = dims
    GNFeedForward(
        FeedForward(de, dropout),
        FeedForward(dn, dropout),
        FeedForward(dg, dropout),
    )
end

FeedForward(d, dropout) = Chain(
    Dense(d => 4d, relu),
    Dense(4d => d),
    Dropout(dropout),
)

function (m::GNFeedForward)(x)
    (
        graphs = x.graphs,
        ef = m.eff(x.ef),
        nf = m.nff(x.nf),
        gf = m.gff(x.gf),
    )
end