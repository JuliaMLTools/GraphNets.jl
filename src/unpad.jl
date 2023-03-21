function unpadef(adj_mat::AbstractMatrix, ef)
    @view ef[:, findall(isone, view(adj_mat, :)), :]
end

function unpadef(adj_mats::AbstractVector, ef)
    padded_adj_mats = padadjmats(adj_mats)
    map(enumerate(eachslice(padded_adj_mats; dims=3))) do (graph_idx,padded_adj_mat)
        idx = findall(isone, view(padded_adj_mat, :))
        view(ef, :, idx, graph_idx)
    end
end

function unpadnf(adj_mats::AbstractVector, nf)
    map(enumerate(adj_mats)) do (graph_idx,adj_mat)
        @view nf[:, 1:size(adj_mat,1), graph_idx]
    end
end

function unpadgf(adj_mats::AbstractMatrix, gf)
    reshape(gf, size(gf,1), :)
end

function unpadgf(adj_mats::AbstractVector, gf)
    collect(eachcol(reshape(gf, size(gf,1), :)))
end