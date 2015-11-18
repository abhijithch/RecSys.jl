using ParallelSparseMatMul

import Base: zero

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
    M::Nullable{SparseVector{AbstractString,Int64}}

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
        R = filter_empty(R)
        inp.R = Nullable(R)
    end

    get(inp.R)
end

function movie_names(inp::Inputs)
    if isnull(inp.M)
        A = read_input(inp.movie_names)
        movie_ids = convert(Array{Int}, A[:,1])
        movie_names = convert(Array{AbstractString}, A[:,2])
        movie_genres = convert(Array{AbstractString}, A[:,3])
        movies = AbstractString[n*" - "*g for (n,g) in zip(movie_names, movie_genres)]
        #movies = [zip(movie_names,movie_genres)...]
        M = SparseVector(maximum(movie_ids), movie_ids, movies)
        inp.M = Nullable(M)
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
    println("reading inputs")
    t1 = time()
    R = ratings(als.inp)
    t2 = time()
    println("read time: $(t2-t1)")
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

function sprows(R::Union{SparseMatrixCSC{Float64,Int64},ParallelSparseMatMul.SharedSparseMatrixCSC{Float64,Int64}}, col::Int64)
    rowstart = R.colptr[col]
    rowend = R.colptr[col+1] - 1
    rows = R.rowval[rowstart:rowend]
    vals = R.nzval[rowstart:rowend]
    rows, vals
end

function update_user(u::Int64)
    c = fetch_compdata()
    U = c.U
    M = c.M
    RT = c.RT
    lambdaI = c.lambdaI

    nzrows, nzvals = sprows(RT, u)
    Mu = M[:, nzrows]
    vec = Mu * nzvals
    mat = (Mu * Mu') + (length(nzrows) * lambdaI)
    U[u,:] = mat \ vec
    nothing
end

function update_movie(m::Int64)
    c = fetch_compdata()
    U = c.U
    M = c.M
    R = c.R
    lambdaI = c.lambdaI

    nzrows, nzvals = sprows(R, m)
    Um = U[nzrows, :]
    Umt = Um'
    vec = Umt * nzvals
    mat = (Umt * Um) + (length(nzrows) * lambdaI)
    M[:,m] = mat \ vec
    nothing
end

function fact(R::SparseMatrixCSC{Float64,Int64}, niters::Int, nfactors::Int64)
    t1 = time()
    println("preparing inputs")
    U, M, R = prep(R, nfactors)
    nusers, nmovies = size(R)

    lambda = 0.065
    lambdaI = lambda * eye(nfactors)

    RT = R'
    t2 = time()
    println("prep time: $(t2-t1)")
    fact_iters(U, M, R, RT, niters, nusers, nmovies, lambdaI)
end

type ComputeData
    U::SharedArray{Float64,2}
    M::SharedArray{Float64,2}
    R::ParallelSparseMatMul.SharedSparseMatrixCSC{Float64,Int64}
    RT::ParallelSparseMatMul.SharedSparseMatrixCSC{Float64,Int64}
    lambdaI::SharedArray{Float64,2}
end

const compdata = ComputeData[]

share_compdata(c::ComputeData) = (push!(compdata, c); nothing)
fetch_compdata() = compdata[1]

function fact_iters(_U::Matrix{Float64}, _M::Matrix{Float64}, _R::SparseMatrixCSC{Float64,Int64}, _RT::SparseMatrixCSC{Float64,Int64},
            niters::Int64, nusers::Int64, nmovies::Int64, _lambdaI::Matrix{Float64})
    t1 = time()
    U = share(_U)
    M = share(_M)
    R = share(_R)
    RT = share(_RT)
    lambdaI = share(_lambdaI)

    c =ComputeData(U, M, R, RT, lambdaI)
    for w in workers()
        remotecall_fetch(share_compdata, w, c)
    end

    for iter in 1:niters
        println("iter $iter users")
        @sync @parallel for u in 1:nusers
            update_user(u)
        end
        println("iter $iter movies")
        @sync @parallel for m in 1:nmovies
            update_movie(m)
        end
    end

    t2 = time()
    println("fact time $(t2-t1)")
    copy(U), copy(M)
end

function rmse(als)
    t1 = time()
    model = get(als.model)
    U = share(model.U)
    M = share(model.M)

    R = ratings(als.inp)
    RT = share(R')

    cumerr = @parallel (+) for user in 1:size(RT, 2)
        Uvec = reshape(U[user, :], 1, size(U, 2))
        nzrows, nzvals = sprows(RT, user)
        predicted = vec(Uvec*M)[nzrows]
        sqrt(sum((predicted .- nzvals) .^ 2) / length(predicted))
    end
    cumerr
end

zero(::Type{AbstractString}) = ""
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
        seen = full(R[user,:])
        seen = find(seen)

        # filter out movies already seen
        println("seen: $seen")
        println(mnames[seen])
    end

    movieids = top[1:count]
    mnames[movieids]
end
