type ALSWR{TP<:Parallelism,TI<:Inputs,TM<:Model}
    inp::TI
    model::Nullable{TM}
    par::TP
end

ALSWR{TP<:Parallelism}(inp::FileSpec, par::TP=ParShmem()) = ALSWR{TP,SharedMemoryInputs,SharedMemoryModel}(SharedMemoryInputs(inp), nothing, par)

function clear(als::ALSWR)
    clear(als.inp)
    isnull(als.model) || clear(get(als.model))
end

function localize!(als::ALSWR)
    localize!(als.inp)
    isnull(als.model) || localize!(get(als.model))
end

function save(als::ALSWR, filename::AbstractString)
    clear(als)
    open(filename, "w") do f
        serialize(f, als)
    end
    nothing
end

function train(als::ALSWR, niters::Int, nfacts::Int64, lambda::Float64=0.065)
    als.model = prep(als.inp, nfacts, lambda)
    fact_iters(als, niters)
    nothing
end

fact_iters(als, niters) = fact_iters(als.par, get(als.model), als.inp, niters)

##
# Training
#
function update_user(u::Int64)
    c = fetch_compdata()
    update_user(u, c.model, c.inp, c.lambdaI)
end

function update_user(u::Int64, model, inp, lambdaI)
    nzrows, nzvals = items_and_ratings(inp, u)
    Pu = getP(model, nzrows)
    vec = Pu * nzvals
    mat = (Pu * Pu') + (length(nzrows) * lambdaI)
    setU(model, u, mat \ vec)
    nothing
end

function update_item(i::Int64)
    c = fetch_compdata()
    update_item(i::Int64, c.model, c.inp, c.lambdaI)
end

function update_item(i::Int64, model, inp, lambdaI)
    nzrows, nzvals = users_and_ratings(inp, i)
    Ui = getU(model, nzrows)
    Uit = Ui'
    vec = Uit * nzvals
    mat = (Uit * Ui) + (length(nzrows) * lambdaI)
    setP(model, i, mat \ vec)
    nothing
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

    testinp = SharedMemoryInputs(testdataset)
    ensure_loaded(testinp; only_items=i_idmap)
    rmse(als, testinp)
end


##
# Recommendation
function _recommend(Uvec, model, rated, i_idmap; count::Int=10)
    top = sortperm(vec(vec_mul_p(model, Uvec)))

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

    # All the items sorted in decreasing order of rating.
    Uvec = reshape(getU(model, user), 1, nfactors(model))
    rated = unrated ? all_items_rated(als.inp, user) : Int64[]

    _recommend(Uvec, model, rated, i_idmap; count=count)
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
    Rvec = reshape(user_ratings[i_idmap], 1, length(i_idmap))
    #Rvec = reshape(user_ratings, 1, length(i_idmap))
    Uvec = vec_mul_pinv(model, Rvec)

    _recommend(Uvec, model, find(Rvec), i_idmap; count=count)
end



##
# Shared memory parallelism
type ComputeData{TM<:Model,TI<:Inputs}
    model::TM
    inp::TI
    lambdaI::ModelFactor # store directly, avoid null check on every iteration
end

const compdata = ComputeData[]

share_compdata(c::ComputeData) = (push!(compdata, c); nothing)
fetch_compdata() = compdata[1]
noop(args...) = nothing

function fact_iters{TP<:ParShmem,TM<:Model,TI<:Inputs}(::TP, model::TM, inp::TI, niters::Int64)
    t1 = time()
    share!(model)
    share!(inp)

    c = ComputeData(model, inp, get(model.lambdaI))
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

    localize!(model)

    t2 = time()
    logmsg("fact time $(t2-t1)")
    nothing
end

function rmse{TP<:ParShmem,TI<:Inputs}(als::ALSWR{TP}, inp::TI)
    t1 = time()

    model = get(als.model)
    share!(model)
    share!(inp)

    cumerr = @parallel (.+) for user in 1:nusers(inp)
        Uvec = reshape(getU(model,user), 1, nfactors(model))
        nzrows, nzvals = items_and_ratings(inp, user)
        predicted = vec(vec_mul_p(model, Uvec))[nzrows]
        [sum((predicted .- nzvals) .^ 2), length(predicted)]
    end
    localize!(model)
    logmsg("rmse time $(time()-t1)")
    sqrt(cumerr[1]/cumerr[2])
end


##
# Thread parallelism
if (Base.VERSION >= v"0.5.0-")

function thread_update_item{TM<:Model,TI<:Inputs}(model::TM, inp::TI, ni::Int64, lambdaI::Matrix{Float64})
    @threads for i in Int64(1):ni
        update_item(i, model, inp, lambdaI)
    end
    nothing
end

function thread_update_user{TM<:Model,TI<:Inputs}(model::TM, inp::TI, nu::Int64, lambdaI::Matrix{Float64})
    @threads for u in Int64(1):nu
        update_user(u, model, inp, lambdaI)
    end
    nothing
end
function fact_iters{TP<:ParThread,TM<:Model,TI<:Inputs}(::TP, model::TM, inp::TI, niters::Int64)
    t1 = time()

    nu = nusers(inp)
    ni = nitems(inp)
    lambdaI = get(model.lambdaI)
    for iter in 1:niters
        logmsg("begin iteration $iter")
        # gc is not threadsafe yet. issue #10317
        gc_enable(false)
        thread_update_user(model, inp, nu, lambdaI)
        gc_enable(true)
        gc()
        gc_enable(false)
        logmsg("\tusers")
        thread_update_item(model, inp, ni, lambdaI)
        gc_enable(true)
        gc()
        logmsg("\titems")
    end

    t2 = time()
    logmsg("fact time $(t2-t1)")
    nothing
end

function rmse{TP<:ParThread,TI<:Inputs}(als::ALSWR{TP}, inp::TI)
    t1 = time()

    model = get(als.model)
    errs = zeros(nthreads())
    lengths = zeros(Int, nthreads())

    pos = 1
    NF = nfactors(model)
    N2 = nusers(als.inp)
    while pos < N2
        endpos = min(pos+10000, N2)
        gc_enable(false)
        @threads for user in pos:endpos
            Uvec = reshape(getU(model, user), 1, NF)
            nzrows, nzvals = items_and_ratings(inp, user)
            predicted = vec(vec_mul_p(model, Uvec))[nzrows]

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
