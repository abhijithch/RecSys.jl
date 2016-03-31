using RecSys
using Blobs
#include("/home/tan/Work/julia/packages/Blobs/examples/matrix.jl")
using RecSys.MatrixBlobs

function split_sparse(S, chunkmax, metadir)
    isdir(metadir) || mkdir(metadir)
    spblobs = SparseMatBlobs(Float64, Int64, metadir)
    chunknum = 1
    count = 1
    colstart = 1
    nzval = S.nzval
    rowval = S.rowval
    for col in 1:size(S,2)
        npos = S.colptr[col+1]
        if (npos >= (count + chunkmax)) || (col == size(S,2))
            print("\tchunk $chunknum ... ")
            append!(spblobs, S[:, colstart:col])
            colstart = col+1
            count = npos
            chunknum += 1
            println("done")
        end
    end
    Blobs.save(spblobs)
    nothing
end

function splitall(inp::DlmFile, output_path::AbstractString, nsplits::Int)
    println("reading inputs...")
    ratings = RecSys.read_input(inp)

    users   = convert(Vector{Int64},   ratings[:,1]);
    items   = convert(Vector{Int64},   ratings[:,2]);
    ratings = convert(Vector{Float64}, ratings[:,3]);
    R = sparse(users, items, ratings);
    splitall(R, output_path, nsplits)
end

function splitall(R::SparseMatrixCSC, output_path::AbstractString, nsplits::Int)
    nratings = nnz(R)
    println("$nratings ratings in $(size(R)) sized sparse matrix")
    println("removing empty...")
    R, non_empty_items, non_empty_users = RecSys.filter_empty(R)
    nratings = nnz(R)
    println("$nratings ratings in $(size(R)) sized sparse matrix")

    nsplits_u = round(Int, nratings/nsplits)
    nsplits_i = round(Int, nratings/nsplits)

    println("splitting R itemwise at $nsplits_i items...")
    split_sparse(R, nsplits_i, joinpath(output_path, "R_itemwise"))
    RT = R'
    println("splitting RT userwise at $nsplits_u users...")
    split_sparse(RT, nsplits_u, joinpath(output_path, "RT_userwise"))
    nothing
end

function split_movielens(dataset_path = "/data/Work/datasets/movielens/ml-20m")
    ratings_file = DlmFile(joinpath(dataset_path, "ratings.csv"); dlm=',', header=true)
    splitall(ratings_file, joinpath(dataset_path, "splits"), 10)
end

function split_lastfm(dataset_path = "/data/Work/datasets/last_fm_music_recommendation/profiledata_06-May-2005")

    function read_artist_map(artist_map::FileSpec)
        t1 = time()
        RecSys.@logmsg("reading artist map")
        A = read_input(artist_map)
        valid = map(x->isa(x, Integer), A)
        valid = valid[:,1] & valid[:,2]
        Avalid = convert(Matrix{Int64}, A[valid, :])

        amap = Dict{Int64,Int64}()
        for idx in 1:size(Avalid,1)
            bad_id = Avalid[idx, 1]
            good_id = Avalid[idx, 2]
            amap[bad_id] = good_id
        end
        RecSys.@logmsg("read artist map in $(time()-t1) secs")
        amap
    end

    function read_trainingset(trainingset::FileSpec, amap::Dict{Int64,Int64})
        t1 = time()
        RecSys.@logmsg("reading trainingset")
        T = read_input(trainingset)
        for idx in 1:size(T,1)
            artist_id = T[idx,2]
            if artist_id in keys(amap)
                T[idx,2] = amap[artist_id]
            end
        end
        users   = convert(Vector{Int64},   T[:,1])
        artists = convert(Vector{Int64},   T[:,2])
        ratings = convert(Vector{Float64}, T[:,3])
        S = sparse(users, artists, ratings)
        RecSys.@logmsg("read trainingset in $(time()-t1) secs")
        S
    end

    amap = read_artist_map(DlmFile(joinpath(dataset_path, "artist_alias.txt")))
    T = read_trainingset(DlmFile(joinpath(dataset_path, "user_artist_data.txt")), amap)

    println("randomizing items to remove skew")
    T = T[:, randperm(size(T,2))]

    splitall(T, joinpath(dataset_path, "splits"), 20)
end

function load_splits(dataset_path = "/data/Work/datasets/last_fm_music_recommendation/profiledata_06-May-2005/splits")
    sp = SparseMatBlobs(joinpath(dataset_path, "R_itemwise"))
    for idx in 1:length(sp.splits)
        p = sp.splits[idx]
        r = p.first
        part, _r = Blobs.load(sp, first(r))
        RecSys.@logmsg("got part of size: $(size(part)), with r: $r, _r:$_r")
    end
    println("finished")
end

##
# an easy way to generate somewhat relevant test data is to take an existing dataset and replicate
# items and users based on existing data.
function Blobs.append!{Tv,Ti}(sp::SparseMatBlobs{Tv,Ti}, blob::Blob)
    S = blob.data.value
    m,n = size(S)
    if isempty(sp.splits)
        sp.sz = (m, n)
        idxrange = 1:n
    else
        (sp.sz[1] == m) || throw(BoundsError("SparseMatBlobs $(sp.sz)", (m,n)))
        old_n = sp.sz[2]
        idxrange = (old_n+1):(old_n+n)
        sp.sz = (m, old_n+n)
    end

    push!(sp.splits, idxrange => blob.id)
    RecSys.@logmsg("appending blob $(blob.id) of size: $(size(S)) for idxrange: $idxrange, sersz: $(blob.metadata.size)")
    blob
end

function generate_test_data(setname::AbstractString, generated_data_path, original_data_path, mul_factor)
    sp1 = SparseMatBlobs(joinpath(original_data_path, setname))
    metapath = joinpath(generated_data_path, setname)
    sp2 = SparseMatBlobs(Float64, Int64, metapath)
    isdir(metapath) || mkdir(metapath)
    for idx in 1:length(sp1.splits)
        p = sp1.splits[idx]
        r = p.first
        part, _r = Blobs.load(sp1, first(r))
        RecSys.@logmsg("got part of size: $(size(part)), with r: $r, _r:$_r")
        part_out = part
        for x in 1:(mul_factor-1)
            part_out = vcat(part_out, part)
        end
        blob = append!(sp2, part_out)
        for x in 1:(mul_factor-1)
            append!(sp2, blob)
            RecSys.@logmsg("generated part of size: $(size(part_out))")
        end
    end
    Blobs.save(sp2)
end

function generate_test_data(generated_data_path = "/data/Work/datasets/last_fm_music_recommendation/profiledata_06-May-2005/splits2",
        original_data_path = "/data/Work/datasets/last_fm_music_recommendation/profiledata_06-May-2005/splits",
        mul_factor=2)
    generate_test_data("R_itemwise", generated_data_path, original_data_path, mul_factor)
    generate_test_data("RT_userwise", generated_data_path, original_data_path, mul_factor)
    println("finished")
end
