# Note:
# Filtering out causes the item and user ids to change.
# We need to keep a mapping to be able to match in the recommend step.
function filter_empty(R::RatingMatrix)
    U = sum(R, 2)
    non_empty_users = find(U)
    R = R[non_empty_users, :]

    R = R'
    I = sum(R, 2)
    non_empty_items = find(I)
    R = R[non_empty_items, :]

    R', non_empty_items, non_empty_users
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
    I::Matrix{Float64}
    lambda::Float64
end

type ALSWR
    inp::Inputs
    model::Nullable{Model}

    function ALSWR(inp::FileSpec)
        new(Inputs(inp), nothing)
    end
end

function ratings(als::ALSWR)
    inp = als.inp
    if isnull(inp.R)
        logmsg("loading inputs...")
        t1 = time()
        A = read_input(inp.ratings_file)

        # separate the columns and make them of appropriate types
        users   = convert(Vector{Int},     A[:,1])
        items   = convert(Vector{Int},     A[:,2])
        ratings = convert(Vector{Float64}, A[:,3])

        # create a sparse matrix
        R = sparse(users, items, ratings)
        R, item_idmap, user_idmap = filter_empty(R)
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
    U, I = fact(R, niters, nfactors, lambda)
    model = Model(U, I, lambda)
    als.model = Nullable(model)
    nothing
end

##
# Training
#
function prep(R::RatingMatrix, nfactors::Int)
    nusers, nitems = size(R)

    U = zeros(nusers, nfactors)
    I = rand(nfactors, nitems)
    for idx in 1:nitems
        I[1,idx] = mean(nonzeros(R[:,idx]))
    end

    U, I, R
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
    U = c.U
    I = c.I
    RT = c.RT
    lambdaI = c.lambdaI

    nzrows, nzvals = sprows(RT, u)
    Iu = I[:, nzrows]
    vec = Iu * nzvals
    mat = (Iu * Iu') + (length(nzrows) * lambdaI)
    U[u,:] = mat \ vec
    nothing
end

function update_item(i::Int64)
    c = fetch_compdata()
    U = c.U
    I = c.I
    R = c.R
    lambdaI = c.lambdaI

    nzrows, nzvals = sprows(R, i)
    Ui = U[nzrows, :]
    Uit = Ui'
    vec = Uit * nzvals
    mat = (Uit * Ui) + (length(nzrows) * lambdaI)
    I[:,i] = mat \ vec
    nothing
end

function fact(R::RatingMatrix, niters::Int, nfactors::Int64, lambda::Float64)
    t1 = time()
    logmsg("preparing inputs...")
    U, I, R = prep(R, nfactors)
    nusers, nitems = size(R)

    lambdaI = lambda * eye(nfactors)

    RT = R'
    t2 = time()
    logmsg("prep time: $(t2-t1)")
    fact_iters(U, I, R, RT, niters, nusers, nitems, lambdaI)
end

type ComputeData
    U::SharedArray{Float64,2}
    I::SharedArray{Float64,2}
    R::SharedRatingMatrix
    RT::SharedRatingMatrix
    lambdaI::SharedArray{Float64,2}
end

const compdata = ComputeData[]

share_compdata(c::ComputeData) = (push!(compdata, c); nothing)
fetch_compdata() = compdata[1]
noop(args...) = nothing

function fact_iters(_U::Matrix{Float64}, _I::Matrix{Float64}, _R::RatingMatrix, _RT::RatingMatrix,
            niters::Int64, nusers::Int64, nitems::Int64, _lambdaI::Matrix{Float64})
    t1 = time()
    U = share(_U)
    I = share(_I)
    R = share(_R)
    RT = share(_RT)
    lambdaI = share(_lambdaI)

    c = ComputeData(U, I, R, RT, lambdaI)
    for w in workers()
        remotecall_fetch(share_compdata, w, c)
    end

    for iter in 1:niters
        logmsg("iter $iter users")
        @parallel (noop) for u in 1:nusers
            update_user(u)
        end
        logmsg("iter $iter items")
        @parallel (noop) for i in 1:nitems
            update_item(i)
        end
    end

    t2 = time()
    logmsg("fact time $(t2-t1)")

    copy(U), copy(I)
end

function rmse(als::ALSWR)
    t1 = time()
    model = get(als.model)
    U = share(model.U)
    I = share(model.I)

    R, _i_idmap, _u_idmap = ratings(als)
    RT = share(R')

    cumerr = @parallel (.+) for user in 1:size(RT, 2)
        Uvec = reshape(U[user, :], 1, size(U, 2))
        nzrows, nzvals = sprows(RT, user)
        predicted = vec(Uvec*I)[nzrows]
        [sum((predicted .- nzvals) .^ 2), length(predicted)]
    end
    sqrt(cumerr[1]/cumerr[2])
end

function recommend(als::ALSWR, user::Int; unrated::Bool=true, count::Int=10)
    R, item_idmap, user_idmap = ratings(als)
    (user in user_idmap) || (return (Int[], Int[], 0))
    user = findfirst(user_idmap, user)

    model = get(als.model)
    U = model.U
    I = model.I

    # All the items sorted in decreasing order of rating.
    Uvec = reshape(U[user, :], 1, size(U, 2))
    top = sortperm(vec(Uvec*I))
    rated = unrated ? find(full(R[user,:])) : Int64[]

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
