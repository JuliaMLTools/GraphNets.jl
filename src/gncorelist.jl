"""
    GNCoreList(input_dim, num_heads; head_size=(input_dim ÷ num_heads), dropout=0)

Initializes an instance of the **`GNCoreList`** type, representing a sequence of GraphNet core blocks composed together.

The following keyword arguments are supported:
- `dropout` (Defaults to 0)

A **`GNCoreList`** instance accepts an input array **`x`** of dimensions (C, T, B) and outputs an array of dimensions (HS, T, B). "C" is the channel size (embedding dimension). "T" is the block size (number of input tokens). "B" is the batch size.

## Examples:

```julia
dims = CE, CN, CG = (10,5,3) # Channel dims (edge, node, graph) for each core
N, B = 3, 2 # N = Node count, B = Batch size
PE = N^2 # PE = Padded Edge Count
adj_mats = [rand(0:1, N, N) for _ in 1:B] # Randomize adjacency matrices
x = (
    graphs = GNGraphBatch(adj_mats),
    ef = rand(Float32, CE, PE, B),
    nf = rand(Float32, CN, N, B), 
    gf = rand(Float32, CG, 1, B)
)
core_list = GNCoreList([GNCore(dims), GNCore(dims)])
y = core_list(x)
@test size(y.ef) == (CE, PE, B)
@test size(y.nf) == (CN, N, B)
@test size(y.gf) == (CG, 1, B)
```
"""
struct GNCoreList
    list
end

Functors.@functor GNCoreList

function (m::GNCoreList)(x)
    foldl((i,fn)->fn(i), m.list; init=x)
end