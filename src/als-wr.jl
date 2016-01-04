# Note:
# Filtering out causes the item and user ids to change.
# We need to keep a mapping to be able to match in the recommend step.
function filter_empty(R::RatingMatrix; only_items::Vector{Int64}=Int64[])
    if !isempty(only_items)
        max_num_items = maximum(only_items)
        if size(R, 2) < max_num_items
            RI, RJ, RV = findnz(R)
            R = sparse(RI, RJ, RV, size(R, 1), max_num_items)
        end
        R = R'
        R = R[only_items, :]
        R = R'
        non_empty_items = only_items
    end

    U = sum(R, 2)
    non_empty_users = find(U)
    R = R[non_empty_users, :]

    if isempty(only_items)
        R = R'
        P = sum(R, 2)
        non_empty_items = find(P)
        R = R[non_empty_items, :]
        R = R'
    end

    R, non_empty_items, non_empty_users
end

type Inputs
    ratings_file::FileSpec
    R::Nullable{RatingMatrix}
    item_idmap::Nullable{Vector{Int64}}
    user_idmap::Nullable{Vector{Int64}}

    function Inputs(file::FileSpec)
        new(file, nothing, nothing, nothing)
    end
end

type Model
    U::Matrix{Float64}
    P::Matrix{Float64}
    lambda::Float64
end

type ALSWR{T<:Parallelism}
    inp::Inputs
    model::Nullable{Model}
    par::T
end

ALSWR{T<:Parallelism}(inp::FileSpec, par::T=ParShmem()) = ALSWR{T}(Inputs(inp), nothing, par)

ratings(als::ALSWR) = ratings(als.inp)
function ratings(inp::Inputs; only_items::Vector{Int64}=Int64[])
    if isnull(inp.R)
        logmsg("loading inputs...")
        t1 = time()
        A = read_input(inp.ratings_file)

        if isa(A, SparseMatrixCSC)
            R = convert(SparseMatrixCSC{Float64,Int64}, A)
        else
            # separate the columns and make them of appropriate types
            users   = convert(Vector{Int64},   A[:,1])
            items   = convert(Vector{Int64},   A[:,2])
            ratings = convert(Vector{Float64}, A[:,3])

            # create a sparse matrix
            R = sparse(users, items, ratings)
        end

        R, item_idmap, user_idmap = filter_empty(R; only_items=only_items)
        inp.R = Nullable(R)
        inp.item_idmap = Nullable(item_idmap)
        inp.user_idmap = Nullable(user_idmap)
        t2 = time()
        logmsg("time to load inputs: $(t2-t1) secs")
    end

    get(inp.R), get(inp.item_idmap), get(inp.user_idmap)
end

function train(als::ALSWR, niters::Int, nfactors::Int64, lambda::Float64=0.065)
    R, _i_idmap, _u_idmap = ratings(als)
    U, P = fact(als.par, R, niters, nfactors, lambda)
    model = Model(U, P, lambda)
    als.model = Nullable(model)
    nothing
end


##
# Training
#
function prep(R::RatingMatrix, nfactors::Int)
    nusers, nitems = size(R)

    U = zeros(nusers, nfactors)
    P = rand(nfactors, nitems)
    for idx in 1:nitems
        P[1,idx] = mean(nonzeros(R[:,idx]))
    end

    U, P, R
end

function sprows(R::Union{RatingMatrix,SharedRatingMatrix}, col::Int64)
    rowstart = R.colptr[col]
    rowend = R.colptr[col+1] - 1
    # use subarray?
    rows = R.rowval[rowstart:rowend]
    vals = R.nzval[rowstart:rowend]
    rows, vals
end

function update_user(u::Int64)
    c = fetch_compdata()
    update_user(u, c.U, c.P, c.RT, c.lambdaI)
end

function update_user(u::Int64, U, P, RT, lambdaI)
    nzrows, nzvals = sprows(RT, u)
    Pu = P[:, nzrows]
    vec = Pu * nzvals
    mat = (Pu * Pu') + (length(nzrows) * lambdaI)
    U[u,:] = mat \ vec
    nothing
end

function update_item(i::Int64)
    c = fetch_compdata()
    update_item(i::Int64, c.U, c.P, c.R, c.lambdaI)
end

function update_item(i::Int64, U, P, R, lambdaI)
    nzrows, nzvals = sprows(R, i)
    Ui = U[nzrows, :]
    Uit = Ui'
    vec = Uit * nzvals
    mat = (Uit * Ui) + (length(nzrows) * lambdaI)
    P[:,i] = mat \ vec
    nothing
end

function fact(par, R::RatingMatrix, niters::Int, nfactors::Int64, lambda::Float64)
    t1 = time()
    logmsg("preparing inputs...")
    U, P, R = prep(R, nfactors)
    nusers, nitems = size(R)

    lambdaI = lambda * eye(nfactors)

    RT = R'
    t2 = time()
    logmsg("prep time: $(t2-t1)")
    fact_iters(par, U, P, R, RT, niters, nusers, nitems, lambdaI)
end


##
# Validation
function rmse(als::ALSWR)
    R, _i_idmap, _u_idmap = ratings(als)
    rmse(als, R)
end

function rmse(als::ALSWR, testdataset::FileSpec)
    _R, i_idmap, _u_idmap = ratings(als)
    R, _i_idmap, _u_idmap = ratings(Inputs(testdataset); only_items=i_idmap)
    rmse(als, R)
end


##
# Recommendation
function _recommend(Uvec, P, rated, item_idmap; count::Int=10)
    top = sortperm(vec(Uvec*P))

    recommended = Int64[]
    idx = 1
    while length(recommended) < count && length(top) >= idx
        item_id = top[idx]
        (item_id in rated) || push!(recommended, item_idmap[item_id])
        idx += 1
    end
    nexcl = idx - count - 1

    mapped_rated = Int64[item_idmap[id] for id in rated]
    recommended, mapped_rated, nexcl
end

function recommend(als::ALSWR, user::Int; unrated::Bool=true, count::Int=10)
    R, item_idmap, user_idmap = ratings(als)
    (user in user_idmap) || (return (Int[], Int[], 0))
    user = findfirst(user_idmap, user)

    model = get(als.model)
    U = model.U
    P = model.P

    # All the items sorted in decreasing order of rating.
    Uvec = reshape(U[user, :], 1, size(U, 2))
    rated = unrated ? find(full(R[user,:])) : Int64[]

    _recommend(Uvec, P, rated, item_idmap; count=count)
end

function recommend(als::ALSWR, user_ratings::SparseVector{Float64,Int64}; unrated::Bool=true, count::Int=10)
    # R = U * P
    # given a new row in R, figure out the corresponding row in U
    #
    # Rvec = Uvec * P
    # Uvec = Rvec * Pinv
    #
    # since: I = (P * Pt) * inv(P * Pt)
    # Pinv = Pt * inv(P * Pt)
    #
    # Uvec = Rvec * (Pt * inv(P * Pt))
    _R, item_idmap, _user_idmap = ratings(als)
    model = get(als.model)
    P = model.P
    PT = P'
    Pinv = PT * inv(P * PT)
    Rvec = reshape(user_ratings[item_idmap], 1, length(item_idmap))
    #Rvec = reshape(user_ratings, 1, length(item_idmap))
    Uvec = Rvec * Pinv

    _recommend(Uvec, P, find(Rvec), item_idmap; count=count)
end



##
# Shared memory parallelism
type ComputeData
    U::SharedArray{Float64,2}
    P::SharedArray{Float64,2}
    R::SharedRatingMatrix
    RT::SharedRatingMatrix
    lambdaI::SharedArray{Float64,2}
end

const compdata = ComputeData[]

share_compdata(c::ComputeData) = (push!(compdata, c); nothing)
fetch_compdata() = compdata[1]
noop(args...) = nothing

function fact_iters{T<:ParShmem}(::T, _U::Matrix{Float64}, _P::Matrix{Float64}, _R::RatingMatrix, _RT::RatingMatrix,
            niters::Int64, nusers::Int64, nitems::Int64, _lambdaI::Matrix{Float64})
    t1 = time()
    U = share(_U)
    P = share(_P)
    R = share(_R)
    RT = share(_RT)
    lambdaI = share(_lambdaI)

    c = ComputeData(U, P, R, RT, lambdaI)
    for w in workers()
        remotecall_fetch(share_compdata, w, c)
    end

    for iter in 1:niters
        logmsg("begin iteration $iter")
        pmap(update_user, 1:nusers)
        #@parallel (noop) for u in 1:nusers
        #    update_user(u)
        #end
        logmsg("\tusers")
        pmap(update_item, 1:nitems)
        #@parallel (noop) for i in 1:nitems
        #    update_item(i)
        #end
        logmsg("\titems")
    end

    t2 = time()
    logmsg("fact time $(t2-t1)")

    copy(U), copy(P)
end

function rmse{T<:ParShmem}(als::ALSWR{T}, R::RatingMatrix)
    t1 = time()

    model = get(als.model)
    U = share(model.U)
    P = share(model.P)
    RT = share(R')

    cumerr = @parallel (.+) for user in 1:size(RT, 2)
        Uvec = reshape(U[user, :], 1, size(U, 2))
        nzrows, nzvals = sprows(RT, user)
        predicted = vec(Uvec*P)[nzrows]
        [sum((predicted .- nzvals) .^ 2), length(predicted)]
    end
    logmsg("rmse time $(time()-t1)")
    sqrt(cumerr[1]/cumerr[2])
end


##
# Thread parallelism
if (Base.VERSION >= v"0.5.0-")

function thread_update_item(U::Matrix{Float64}, P::Matrix{Float64}, R::RatingMatrix, nitems::Int64, lambdaI::Matrix{Float64})
    @threads for i in Int64(1):nitems
        update_item(i, U, P, R, lambdaI)
    end
    nothing
end

function thread_update_user(U::Matrix{Float64}, P::Matrix{Float64}, RT::RatingMatrix, nusers::Int64, lambdaI::Matrix{Float64})
    @threads for u in Int64(1):nusers
        update_user(u, U, P, RT, lambdaI)
    end
    nothing
end
function fact_iters{T<:ParThread}(::T, U::Matrix{Float64}, P::Matrix{Float64}, R::RatingMatrix, RT::RatingMatrix,
            niters::Int64, nusers::Int64, nitems::Int64, lambdaI::Matrix{Float64})
    t1 = time()

    for iter in 1:niters
        logmsg("begin iteration $iter")
        # gc is not threadsafe yet. issue #10317
        gc_enable(false)
        thread_update_user(U, P, RT, nusers, lambdaI)
        gc_enable(true)
        gc()
        gc_enable(false)
        logmsg("\tusers")
        thread_update_item(U, P, R, nitems, lambdaI)
        gc_enable(true)
        gc()
        logmsg("\titems")
    end

    t2 = time()
    logmsg("fact time $(t2-t1)")

    U, P
end

function rmse{T<:ParThread}(als::ALSWR{T}, R::RatingMatrix)
    t1 = time()

    model = get(als.model)
    U = model.U
    P = model.P
    RT = R'

    errs = zeros(nthreads())
    lengths = zeros(Int, nthreads())

    pos = 1
    NU = size(U, 2)
    N2 = size(RT, 2)
    while pos < N2
        endpos = min(pos+10000, N2)
        gc_enable(false)
        @threads for user in pos:endpos
            Uvec = reshape(U[user, :], 1, NU)
            nzrows, nzvals = sprows(RT, user)
            predicted = vec(Uvec*P)[nzrows]

            tid = threadid()
            lengths[tid] += length(predicted)
            errs[tid] += sum((predicted - nzvals) .^ 2)
        end
        gc_enable(true)
        gc()
        pos = endpos + 1
    end
    logmsg("rmse time $(time()-t1)")
    sqrt(sum(errs)/sum(lengths))
end
end
