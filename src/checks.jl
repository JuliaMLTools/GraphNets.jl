function checks(adj_mat::AbstractMatrix, ef, nf, gf)
    checkshapes3d(ef, nf, gf)
    checksamebatchsize3d(ef, nf, gf)
    checknodeedgecounts(adj_mat, ef, nf)
end

function checks(adj_mats::AbstractVector, ef, nf, gf)
    @assert length(adj_mats) > 0
    checksamebatchsize2d(adj_mats, ef, nf, gf)
    # Check shapes
    for adj_mat in adj_mats
        @assert ndims(adj_mat) == 2 # (N, N)
    end
    checkshapes2d(ef, nf, gf)
    # Check node/edge counts
    for (adj_mat, ef_i, nf_i) in zip(adj_mats, ef, nf)
        checknodeedgecounts(adj_mat, ef_i, nf_i)
    end
end

############################
# Check node/edge counts
############################

function checknodeedgecounts(adj_mat, ef, nf)
    num_nodes = size(adj_mat, 1)
    num_edges = length(filter(isone, adj_mat))
    @assert size(ef, 2) == num_edges
    @assert size(nf, 2) == num_nodes "$(size(nf, 2)) != $num_nodes"
end
function checknodeedgecounts(adj_mat, ef, nf::Nothing)
    num_edges = length(filter(isone, adj_mat))
    @assert size(ef, 2) == num_edges
end
function checknodeedgecounts(adj_mat, ef::Nothing, nf)
    num_nodes = size(adj_mat, 1)
    @assert size(nf, 2) == num_nodes
end
checknodeedgecounts(adj_mat, ef::Nothing, nf::Nothing) = nothing


############################
# Check shapes 3D
############################

function checkshapes3d(ef, nf, gf)
    @assert ndims(ef) == ndims(nf) == 3 # (C, T, B)
    @assert ndims(gf) == 2 "$(ndims(gf)) != 2" # (C, B)
end
function checkshapes3d(ef,nf,gf::Nothing)
    @assert ndims(ef) == ndims(nf) == 3 # (C, T, B)
end
function checkshapes3d(ef,nf::Nothing,gf)
    @assert ndims(ef) == 3 # (C, T, B)
    @assert ndims(gf) == 2 # (C, B)
end
function checkshapes3d(ef,nf::Nothing,gf::Nothing)
    @assert ndims(ef) == 3 # (C, T, B)
end
function checkshapes3d(ef::Nothing,nf,gf)
    @assert ndims(nf) == 3 # (C, T, B)
    @assert ndims(gf) == 2 # (C, B)
end
function checkshapes3d(ef::Nothing,nf,gf::Nothing)
    @assert ndims(nf) == 3 # (C, T, B)
end
function checkshapes3d(ef::Nothing,nf::Nothing,gf)
    @assert ndims(gf) == 2 # (C, B)
end


############################
# Check shapes 2D
############################

function checkshapes2d(ef, nf, gf)
    for (ef_i, nf_i, gf_i) in zip(ef, nf, gf)
        @assert ndims(ef_i) == ndims(nf_i) == 2 # (C, T)
        @assert ndims(gf_i) == 1 # (C,)
    end
end
function checkshapes2d(ef, nf, gf::Nothing)
    for (ef_i, nf_i) in zip(ef, nf)
        @assert ndims(ef_i) == ndims(nf_i) == 2 # (C, T)
    end
end
function checkshapes2d(ef, nf::Nothing, gf)
    for (ef_i, gf_i) in zip(ef, gf)
        @assert ndims(ef_i) == 2 # (C, T)
        @assert ndims(gf_i) == 1 # (C,)
    end
end
function checkshapes2d(ef, nf::Nothing, gf::Nothing)
    for ef_i in ef
        @assert ndims(ef_i) == 2 # (C, T)
    end
end
function checkshapes2d(ef::Nothing, nf, gf)
    for (nf_i, gf_i) in zip(nf, gf)
        @assert ndims(nf_i) == 2 # (C, T)
        @assert ndims(gf_i) == 1 # (C,)
    end
end
function checkshapes2d(ef::Nothing, nf, gf::Nothing)
    for nf_i in nf
        @assert ndims(nf_i) == 2 # (C, T)
    end
end
function checkshapes2d(ef::Nothing, nf::Nothing, gf)
    for gf_i in gf
        @assert ndims(gf_i) == 1 # (C,)
    end
end


############################
# Check same batch size 2D
############################

function checksamebatchsize2d(adj_mats, ef, nf, gf)
    @assert length(adj_mats) == length(ef) == length(nf) == length(gf)
end

function checksamebatchsize2d(adj_mats, ef, nf, gf::Nothing)
    @assert length(adj_mats) == length(ef) == length(nf)
end

function checksamebatchsize2d(adj_mats, ef, nf::Nothing, gf)
    @assert length(adj_mats) == length(ef) == length(gf)
end

function checksamebatchsize2d(adj_mats, ef, nf::Nothing, gf::Nothing)
    @assert length(adj_mats) == length(ef)
end

function checksamebatchsize2d(adj_mats, ef::Nothing, nf, gf)
    @assert length(adj_mats) == length(nf) == length(gf)
end

function checksamebatchsize2d(adj_mats, ef::Nothing, nf, gf::Nothing)
    @assert length(adj_mats) == length(nf)
end

function checksamebatchsize2d(adj_mats, ef::Nothing, nf::Nothing, gf)
    @assert length(adj_mats) == length(gf)
end


############################
# Check same batch size 3D
############################

function checksamebatchsize3d(ef, nf, gf)
    @assert size(ef, 3) == size(nf, 3) == size(gf, 2)
end
function checksamebatchsize3d(ef, nf, gf::Nothing)
    @assert size(ef, 3) == size(nf, 3)
end
function checksamebatchsize3d(ef, nf::Nothing, gf)
    @assert size(ef, 3) == size(gf, 2)
end
checksamebatchsize3d(ef, nf::Nothing, gf::Nothing) = nothing
function checksamebatchsize3d(ef::Nothing, nf, gf)
    @assert size(nf, 3) == size(gf, 2)
end
checksamebatchsize3d(ef::Nothing, nf, gf::Nothing) = nothing
checksamebatchsize3d(ef::Nothing, nf::Nothing, gf) = nothing