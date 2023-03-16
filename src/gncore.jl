struct GNCore
    block
    ffwd
    gn1
    gn2
end

Functors.@functor GNCore

"""
GNCore(input_dim; num_heads=1, head_size=(input_dim÷num_heads), dropout=0)

Initializes an instance of the **`GNCore`** type, representing a transformer block.

A **`GNCore`** instance accepts an input array **`x`** of dimensions (C, T, B) and outputs an array of dimensions (HS, T, B). "C" is the channel size (embedding dimension). "T" is the block size (number of input tokens). "B" is the batch size.

The following keyword arguments are supported:
- `mask` (Defaults to nothing. Must be of dimensions (T, T).)

## Examples:

```julia
C,T,B = 8,3,4
block = GNCore(C)
@assert size(block(rand(Float32, C,T,B))) == (C,T,B)
```
"""
function GNCore(dims; dropout=0)
    @assert any(dims .> (0,0,0))
    GNCore(
        GNBlock(dims=>dims; dropout=dropout),
        nothing,
        nothing,
        nothing,
        # GNFeedForward(dims; dropout=dropout),
        # GNGraphNorm(dims),
        # GNGraphNorm(dims),
    )
end

function (m::GNCore)(x)
    m.block(x)
    # x + m.gnlayer(m.gn1(x)) + m.ffwd(m.gn2(x))
end