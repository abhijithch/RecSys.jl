type ALSWR{TP<:Parallelism,TI<:Inputs,TM<:Model}
    inp::TI
    model::Nullable{TM}
    par::TP
end

ALSWR(inp::FileSpec, par::ParShmem) = ALSWR{ParShmem,SharedMemoryInputs,SharedMemoryModel}(SharedMemoryInputs(inp), nothing, par)
ALSWR(user_item_ratings::FileSpec, item_user_ratings::FileSpec, par::ParBlob) = ALSWR{ParBlob,DistInputs,DistModel}(DistInputs(user_item_ratings, item_user_ratings), nothing, par)

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

function update_user(r::UnitRange)
    c = fetch_compdata()
    for u in r
        update_user(u, c.model, c.inp, c.lambdaI)
    end
end

function update_user(u::Int64, model, inp, lambdaI)
    nzrows, nzvals = items_and_ratings(inp, u)
    Pu = getP(model, nzrows)
    vec = Pu * nzvals
    mat = (Pu * Pu') + (length(nzrows) * lambdaI)
    setU(model, u, mat \ vec)
    nothing
end

function update_item(r::UnitRange)
    c = fetch_compdata()
    for i in r
        update_item(i::Int64, c.model, c.inp, c.lambdaI)
    end
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
        real_item_id = isempty(i_idmap) ? item_id : i_idmap[item_id]
        (item_id in rated) || push!(recommended, real_item_id)
        idx += 1
    end
    nexcl = idx - count - 1

    mapped_rated = isempty(i_idmap) ? rated : Int64[i_idmap[id] for id in rated]
    recommended, mapped_rated, nexcl
end

function recommend(als::ALSWR, user::Int; unrated::Bool=true, count::Int=10)
    ensure_loaded(als.inp)
    i_idmap = item_idmap(als.inp)
    u_idmap = user_idmap(als.inp)

    if !isempty(u_idmap)
        (user in u_idmap) || (return (Int[], Int[], 0))
        user = findfirst(u_idmap, user)
    end

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
    mapped_ratings = isempty(i_idmap) ? full(user_ratings) : user_ratings[i_idmap]
    Rvec = reshape(mapped_ratings, 1, length(mapped_ratings))
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

function share_compdata(c::ComputeData)
    push!(compdata, c)
    ensure_loaded(c.inp)
    ensure_loaded(c.model)
    nothing
end
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
    @logmsg("nusers: $nu, nitems: $ni")
    for iter in 1:niters
        @logmsg("begin iteration $iter")
        pmap(update_user, 1:nu)
        #@parallel (noop) for u in 1:nu
        #    update_user(u)
        #end
        @logmsg("\tusers")
        pmap(update_item, 1:ni)
        #@parallel (noop) for i in 1:ni
        #    update_item(i)
        #end
        @logmsg("\titems")
    end

    localize!(model)

    t2 = time()
    @logmsg("fact time $(t2-t1)")
    nothing
end

function rmse{TP<:Union{ParShmem,ParBlob},TI<:Inputs}(als::ALSWR{TP}, inp::TI)
    t1 = time()

    model = get(als.model)
    share!(model)
    share!(inp)
    ensure_loaded(inp)
    ensure_loaded(model)

    cumerr = @parallel (.+) for user in 1:nusers(inp)
        Uvec = reshape(getU(model,user), 1, nfactors(model))
        nzrows, nzvals = items_and_ratings(inp, user)
        predicted = vec(vec_mul_p(model, Uvec))[nzrows]
        [sum((predicted .- nzvals) .^ 2), length(predicted)]
    end
    localize!(model)
    @logmsg("rmse time $(time()-t1)")
    sqrt(cumerr[1]/cumerr[2])
end

##
# Blob based distributed memory parallelism
function train(als::ALSWR{ParBlob,DistInputs,DistModel}, niters::Int, nfacts::Int64, model_dir::AbstractString, max_cache::Int=10, lambda::Float64=0.065)
    als.model = prep(als.inp, nfacts, lambda, model_dir, max_cache)
    fact_iters(als, niters)
    nothing
end

function fact_iters{TP<:ParBlob,TM<:Model,TI<:Inputs}(::TP, model::TM, inp::TI, niters::Int64)
    t1 = time()

    clear(inp)
    share!(model)
    share!(inp)

    c = ComputeData(model, inp, get(model.lambdaI))
    uranges = UnitRange[p.first for p in get(model.U).splits]
    iranges = UnitRange[p.first for p in get(model.P).splits]

    # clear, share the data and load it again (not required, but more efficient)
    clear(model)
    W = workers()
    for w in W
        remotecall_fetch(share_compdata, w, c)
    end
    ensure_loaded(model)
    U = get(model.U)
    P = get(model.P)

    nu = nusers(inp)
    ni = nitems(inp)
    @logmsg("nusers: $nu, nitems: $ni")
    for iter in 1:niters
        @logmsg("begin iteration $iter")
        flush(U, W; callback=false)
        pmap(update_user, uranges)
        save(U, W)
        @logmsg("\tusers")
        flush(P, W; callback=false)
        pmap(update_item, iranges)
        save(P, W)
        @logmsg("\titems")
    end

    localize!(model)

    t2 = time()
    @logmsg("fact time $(t2-t1)")
    nothing
end


##
# Thread parallelism
if (Base.VERSION >= v"0.5.0-")

ALSWR(inp::FileSpec, par::ParThread) = ALSWR{ParThread,SharedMemoryInputs,SharedMemoryModel}(SharedMemoryInputs(inp), nothing, par)

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
        @logmsg("begin iteration $iter")
        # gc is not threadsafe yet. issue #10317
        thread_update_user(model, inp, nu, lambdaI)
        @logmsg("\tusers")
        thread_update_item(model, inp, ni, lambdaI)
        @logmsg("\titems")
    end

    t2 = time()
    @logmsg("fact time $(t2-t1)")
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
        @threads for user in pos:endpos
            Uvec = reshape(getU(model, user), 1, NF)
            nzrows, nzvals = items_and_ratings(inp, user)
            predicted = vec(vec_mul_p(model, Uvec))[nzrows]

            tid = threadid()
            lengths[tid] += length(predicted)
            errs[tid] += sum((predicted - nzvals) .^ 2)
        end
        pos = endpos + 1
    end
    @logmsg("rmse time $(time()-t1)")
    sqrt(sum(errs)/sum(lengths))
end
end
