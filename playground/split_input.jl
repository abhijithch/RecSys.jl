using RecSys

function split_sparse(S, chunkmax, filepfx)
    metafilename = "$(filepfx).meta"
    open(metafilename, "w") do mfile
        chunknum = 1
        count = 1
        colstart = 1
        splits = UnitRange[]
        nzval = S.nzval
        rowval = S.rowval
        for col in 1:size(S,2)
            npos = S.colptr[col+1]
            if (npos >= (count + chunkmax)) || (col == size(S,2))
                print("\tchunk $chunknum ... ")
                cfilename = "$(filepfx).$(chunknum)"
                println(mfile, colstart, ",", col, ",", cfilename)
                RecSys.mmap_csc_save(S[:, colstart:col], cfilename)
                push!(splits, colstart:col)
                colstart = col+1
                count = npos
                chunknum += 1
                println("done")
            end
        end
        println("splits: $splits")
    end
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
    splitall(T, joinpath(dataset_path, "splits"), 20)
end

function load_splits(dataset_path = "/data/Work/datasets/last_fm_music_recommendation/profiledata_06-May-2005/splits")
    cf = RecSys.ChunkedFile(joinpath(dataset_path, "R_itemwise.meta"), UnitRange{Int64}, SparseMatrixCSC{Float64,Int}, 10)
    println(cf)
    nchunks = length(cf.chunks)
    for idx in 1:10
        cid = floor(Int, nchunks*rand()) + 1
        println("fetching from chunk $cid")
        c = cf.chunks[cid]
        key = floor(Int, length(c.keyrange)*rand()) + c.keyrange.start
        println("fetching key $key")
        r,v = RecSys.data(cf, key)
        #println("\tr:$r, v:$v")
    end
    println("finished")
end

function test_dense_splits(dataset_path = "/tmp/test")
    metafilename = joinpath(dataset_path, "mem.meta")
    cfile = DenseMatChunks(metafilename, 1, (10^6,10))
    RecSys.create(cfile)
    cf = RecSys.read_input(cfile)
    for idx in 1:10
        chunk = RecSys.getchunk(cf, idx*10^4)
        A = RecSys.data(chunk, cf.lrucache)
        @assert A.val[1] == 0.0
        fill!(A.val, idx)
    end
    cf = RecSys.read_input(cfile)
    for idx in 1:10
        chunk = RecSys.getchunk(cf, idx*10^4)
        A = RecSys.data(chunk, cf.lrucache)
        @assert A.val[1] == Float64(idx)
        println(A.val[1])
    end
end

##
# an easy way to generate somewhat relevant test data is to take an existing dataset and replicate
# items and users based on existing data.
function generate_test_data(setname::AbstractString, generated_data_path, original_data_path, mul_factor)
    RecSys.@logmsg("generating $setname")
    @assert mul_factor > 1
    incf = RecSys.ChunkedFile(joinpath(original_data_path, "$(setname).meta"), UnitRange{Int64}, SparseMatrixCSC{Float64,Int}, 2)
    outmetaname = joinpath(generated_data_path, "$(setname).meta")

    outkeystart = 1
    outfileidx = 1
    open(outmetaname, "w") do outmeta
        for chunk in incf.chunks
            L = length(chunk.keyrange)
            S = RecSys.data(chunk, incf.lrucache)
            Sout = S
            for x in 1:(mul_factor-1)
                Sout = vcat(Sout, S)
            end
            outfname = joinpath(generated_data_path, "$(setname).$(outfileidx)")
            outfileidx += 1
            RecSys.mmap_csc_save(Sout, outfname)
            for x in 1:mul_factor
                println(outmeta, outkeystart, ",", outkeystart+L-1, ",", outfname)
                RecSys.@logmsg("generated $setname $outkeystart:$(outkeystart+L-1) with size: $(size(Sout)) from size: $(size(S))")
                outkeystart += L
            end
        end
    end
end

function generate_test_data(generated_data_path = "/data/Work/datasets/last_fm_music_recommendation/profiledata_06-May-2005/splits2",
        original_data_path = "/data/Work/datasets/last_fm_music_recommendation/profiledata_06-May-2005/splits",
        mul_factor=2)
    generate_test_data("R_itemwise", generated_data_path, original_data_path, mul_factor)
    generate_test_data("RT_userwise", generated_data_path, original_data_path, mul_factor)
    println("finished")
end
