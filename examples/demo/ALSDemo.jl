using ParallelSparseMatMul

type FileSpec
    name::AbstractString
    dlm::Char
    header::Bool

    function FileSpec(name::AbstractString, dlm::Char=',', header::Bool=false)
        new(name, dlm, header)
    end
end

function read_input(fspec::FileSpec)
    # read file and skip the header
    F = readdlm(fspec.name, fspec.dlm, header=fspec.header)
    fspec.header ? F[1] : F
end

type Inputs
    movie_names::FileSpec
    ratings::FileSpec
    R::Nullable{SparseMatrixCSC{Float64,Int64}}
    M::Nullable{Matrix}

    function Inputs(movie_names::FileSpec, ratings::FileSpec)
        new(movie_names, ratings, nothing, nothing)
    end
end

function ratings(inp::Inputs)
    if isnull(inp.R)
        A = read_input(inp.ratings)

        # separate the columns and make them of appropriate types
        users   = convert(Vector{Int},     A[:,1])
        movies  = convert(Vector{Int},     A[:,2])
        ratings = convert(Vector{Float64}, A[:,3])

        # create a sparse matrix
        R = sparse(users, movies, ratings)
        #R = filter_empty(R)
        inp.R = Nullable(R)
    end

    get(inp.R)
end

function movie_names(inp::Inputs)
    if isnull(inp.M)
        A = read_input(inp.movie_names)
        inp.M = Nullable(A)
    end

    get(inp.M)
end

type Model
    U::Matrix{Float64}
    M::Matrix{Float64}
end

type MovieALSRec
    inp::Inputs
    model::Nullable{Model}

    function MovieALSRec(inp::Inputs)
        new(inp, nothing)
    end
end

function train(als::MovieALSRec, niters::Int, nfactors::Int64)
    R = ratings(als.inp)
    U, M = fact(R, niters, nfactors)
    model = Model(U, M)
    als.model = Nullable(model)
    nothing
end

function save(als::MovieALSRec, filename::AbstractString)
    open(filename, "w") do f
        serialize(f, als)
    end
    nothing
end

function load(filename::AbstractString)
    open(filename, "r") do f
        deserialize(f)
    end
end


##
# Training
#
function filter_empty(R::SparseMatrixCSC{Float64,Int64})
    U = sum(R, 2)
    non_empty_users = find(U)
    R = R[non_empty_users, :]

    M = sum(R, 1)
    non_empty_movies = find(M)
    R[:, non_empty_movies]
end

function prep(R::SparseMatrixCSC{Float64,Int64}, nfactors::Int)
    nusers, nmovies = size(R)

    U = zeros(nusers, nfactors)
    M = rand(nfactors, nmovies)
    for idx in 1:nmovies
        M[1,idx] = mean(nonzeros(R[:,idx]))
    end

    U, M, R
end

function nnz_counts(R::SparseMatrixCSC{Float64,Int64})
    r, c, v = findnz(R)
    S = sparse(r, c, 1)

    nnzM = sum(S, 1)
    nnzU = sum(S, 2)

    nnzU, nnzM
end

function nnz_locs(R::SparseMatrixCSC{Float64,Int64})
    res = Dict{Int64,Any}()
    for idx in 1:size(R,2)
        res[idx] = find(full(R[:,idx]))
    end
    res
end

function update_user(U, M, RT, nzM, u, nnzU, lambdaI)
    nzmu = nzM[u]
    Mu = M[:, nzmu]
    vec = Mu * full(RT[nzmu, u])
    mat = (Mu * Mu') + (nnzU[u] * lambdaI)
    U[u,:] = mat \ vec
    nothing
end

function update_movie(U, M, R, nzU, m, nnzM, lambdaI)
    nzum = nzU[m]
    Um = U[nzum, :]
    Umt = Um'
    vec = Umt * full(R[nzum, m])
    mat = (Umt * Um) + (nnzM[m] * lambdaI)
    M[:,m] = mat \ vec
end

function fact(R::SparseMatrixCSC{Float64,Int64}, niters::Int, nfactors::Int64)
    U, M, R = prep(R, nfactors)
    nusers, nmovies = size(R)
    nnzU, nnzM = nnz_counts(R)   

    lambda = 0.065
    lambdaI = lambda * eye(nfactors)

    RT = R'
    nzU = nnz_locs(R)
    nzM = nnz_locs(RT)
    fact_iters(U, M, R, RT, nzU, nzM, niters, nusers, nmovies, nnzU, nnzM, lambdaI)
end

function fact_iters(_U, _M, _R, _RT, nzU, nzM, niters, nusers, nmovies, nnzU, nnzM, _lambdaI)
    U = share(_U)
    M = share(_M)
    R = share(_R)
    RT = share(_RT)
    lambdaI = share(_lambdaI)

    for iter in 1:niters
        @sync @parallel for u in 1:nusers
            update_user(U, M, RT, nzM, u, nnzU, lambdaI)
        end
        @sync @parallel for m in 1:nmovies
            update_movie(U, M, R, nzU, m, nnzM, lambdaI)
        end
    end

    copy(U), copy(M)
end

function recommend(als::MovieALSRec, user::Int; unseen::Bool=true, count::Int=10)
    model = get(als.model)
    U = model.U
    M = model.M

    # All the movies sorted in decreasing order of rating.
    Uvec = reshape(U[user, :], 1, size(U, 2))
    top = sortperm(vec(Uvec*M))

    mnames = movie_names(als.inp)

    if unseen
        R = ratings(als.inp)
        # movies seen by user
        seen = find(full(R[user,:]))

        # filter out movies already seen
        println("seen:")
        println(mnames[seen, 2])
    end

    mnames[top[1:count, :][:], 2]
end
