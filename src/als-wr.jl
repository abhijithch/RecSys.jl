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

function save(model::ALSWR, filename::AbstractString)
    clear(model.inp)
    open(filename, "w") do f
        serialize(f, model)
    end
    nothing
end

function train(als::ALSWR, niters::Int, nfactors::Int64, lambda::Float64=0.065)
    U, P = fact(als, niters, nfactors, lambda)
    model = Model(U, P, lambda)
    als.model = Nullable(model)
    nothing
end


##
# Training
#
function prep(als::ALSWR, nfactors::Int)
    nu = nusers(als.inp)
    ni = nitems(als.inp)

    U = zeros(nu, nfactors)
    P = rand(nfactors, ni)
    for idx in 1:ni
        P[1,idx] = mean(all_user_ratings(als.inp, idx))
    end

    U, P
end

function update_user(u::Int64)
    c = fetch_compdata()
    update_user(u, c.U, c.P, c.inp, c.lambdaI)
end

function update_user(u::Int64, U, P, inp, lambdaI)
    nzrows, nzvals = items_and_ratings(inp, u)
    Pu = P[:, nzrows]
    vec = Pu * nzvals
    mat = (Pu * Pu') + (length(nzrows) * lambdaI)
    U[u,:] = mat \ vec
    nothing
end

function update_item(i::Int64)
    c = fetch_compdata()
    update_item(i::Int64, c.U, c.P, c.inp, c.lambdaI)
end

function update_item(i::Int64, U, P, inp, lambdaI)
    nzrows, nzvals = users_and_ratings(inp, i)
    Ui = U[nzrows, :]
    Uit = Ui'
    vec = Uit * nzvals
    mat = (Uit * Ui) + (length(nzrows) * lambdaI)
    P[:,i] = mat \ vec
    nothing
end

function fact(als::ALSWR, niters::Int, nfactors::Int64, lambda::Float64)
    ensure_loaded(als.inp)
    t1 = time()
    logmsg("preparing inputs...")
    U, P = prep(als, nfactors)
    lambdaI = lambda * eye(nfactors)
    t2 = time()
    logmsg("prep time: $(t2-t1)")
    fact_iters(als.par, U, P, als.inp, niters, lambdaI)
end


##
# Validation
function rmse(als::ALSWR)
    ensure_loaded(als.inp)
    rmse(als, als.inp)
end

function rmse(als::ALSWR, testdataset::FileSpec)
    ensure_loaded(als.inp)
    i_idmap = item_idmap(als.inp)

    testinp = Inputs(testdataset)
    ensure_loaded(testinp; only_items=i_idmap)
    rmse(als, testinp)
end


##
# Recommendation
function _recommend(Uvec, P, rated, i_idmap; count::Int=10)
    top = sortperm(vec(Uvec*P))

    recommended = Int64[]
    idx = 1
    while length(recommended) < count && length(top) >= idx
        item_id = top[idx]
        (item_id in rated) || push!(recommended, i_idmap[item_id])
        idx += 1
    end
    nexcl = idx - count - 1

    mapped_rated = Int64[i_idmap[id] for id in rated]
    recommended, mapped_rated, nexcl
end

function recommend(als::ALSWR, user::Int; unrated::Bool=true, count::Int=10)
    ensure_loaded(als.inp)
    i_idmap = item_idmap(als.inp)
    u_idmap = user_idmap(als.inp)

    (user in u_idmap) || (return (Int[], Int[], 0))
    user = findfirst(u_idmap, user)

    model = get(als.model)
    U = model.U
    P = model.P

    # All the items sorted in decreasing order of rating.
    Uvec = reshape(U[user, :], 1, size(U, 2))
    rated = unrated ? all_items_rated(als.inp, user) : Int64[]

    _recommend(Uvec, P, rated, i_idmap; count=count)
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
    ensure_loaded(als.inp)
    i_idmap = item_idmap(als.inp)
    model = get(als.model)
    P = model.P
    PT = P'
    Pinv = PT * inv(P * PT)
    Rvec = reshape(user_ratings[i_idmap], 1, length(i_idmap))
    #Rvec = reshape(user_ratings, 1, length(i_idmap))
    Uvec = Rvec * Pinv

    _recommend(Uvec, P, find(Rvec), i_idmap; count=count)
end



##
# Shared memory parallelism
type ComputeData
    U::SharedArray{Float64,2}
    P::SharedArray{Float64,2}
    inp::Inputs
    lambdaI::SharedArray{Float64,2}
end

const compdata = ComputeData[]

share_compdata(c::ComputeData) = (push!(compdata, c); nothing)
fetch_compdata() = compdata[1]
noop(args...) = nothing

function fact_iters{T<:ParShmem}(::T, _U::Matrix{Float64}, _P::Matrix{Float64}, inp::Inputs, niters::Int64, _lambdaI::Matrix{Float64})
    t1 = time()
    U = share(_U)
    P = share(_P)
    lambdaI = share(_lambdaI)
    share(inp)

    c = ComputeData(U, P, inp, lambdaI)
    for w in workers()
        remotecall_fetch(share_compdata, w, c)
    end

    nu = nusers(inp)
    ni = nitems(inp)
    for iter in 1:niters
        logmsg("begin iteration $iter")
        pmap(update_user, 1:nu)
        #@parallel (noop) for u in 1:nu
        #    update_user(u)
        #end
        logmsg("\tusers")
        pmap(update_item, 1:ni)
        #@parallel (noop) for i in 1:ni
        #    update_item(i)
        #end
        logmsg("\titems")
    end

    t2 = time()
    logmsg("fact time $(t2-t1)")

    copy(U), copy(P)
end

function rmse{T<:ParShmem}(als::ALSWR{T}, inp::Inputs)
    t1 = time()

    model = get(als.model)
    U = share(model.U)
    P = share(model.P)
    share(inp)

    cumerr = @parallel (.+) for user in 1:nusers(inp)
        Uvec = reshape(U[user, :], 1, size(U, 2))
        nzrows, nzvals = items_and_ratings(inp, user)
        predicted = vec(Uvec*P)[nzrows]
        [sum((predicted .- nzvals) .^ 2), length(predicted)]
    end
    logmsg("rmse time $(time()-t1)")
    sqrt(cumerr[1]/cumerr[2])
end


##
# Thread parallelism
if (Base.VERSION >= v"0.5.0-")

function thread_update_item(U::Matrix{Float64}, P::Matrix{Float64}, inp::Inputs, ni::Int64, lambdaI::Matrix{Float64})
    @threads for i in Int64(1):ni
        update_item(i, U, P, inp, lambdaI)
    end
    nothing
end

function thread_update_user(U::Matrix{Float64}, P::Matrix{Float64}, inp::Inputs, nu::Int64, lambdaI::Matrix{Float64})
    @threads for u in Int64(1):nu
        update_user(u, U, P, inp, lambdaI)
    end
    nothing
end
function fact_iters{T<:ParThread}(::T, U::Matrix{Float64}, P::Matrix{Float64}, inp::Inputs, niters::Int64, lambdaI::Matrix{Float64})
    t1 = time()

    nu = nusers(inp)
    ni = nitems(inp)
    for iter in 1:niters
        logmsg("begin iteration $iter")
        # gc is not threadsafe yet. issue #10317
        gc_enable(false)
        thread_update_user(U, P, inp, nu, lambdaI)
        gc_enable(true)
        gc()
        gc_enable(false)
        logmsg("\tusers")
        thread_update_item(U, P, inp, ni, lambdaI)
        gc_enable(true)
        gc()
        logmsg("\titems")
    end

    t2 = time()
    logmsg("fact time $(t2-t1)")

    U, P
end

function rmse{T<:ParThread}(als::ALSWR{T}, inp::Inputs)
    t1 = time()

    model = get(als.model)
    U = model.U
    P = model.P

    errs = zeros(nthreads())
    lengths = zeros(Int, nthreads())

    pos = 1
    NU = size(U, 2)
    N2 = nusers(als.inp)
    while pos < N2
        endpos = min(pos+10000, N2)
        gc_enable(false)
        @threads for user in pos:endpos
            Uvec = reshape(U[user, :], 1, NU)
            nzrows, nzvals = items_and_ratings(inp, user)
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
