# In distributed memory mode, the model may be too large to load into memory on
# a single node. It is therefore split by one of its dimensions into chunks.
#
# All matrix operations are done on the node where required, just that not all
# chunks are loaded at once at any time. This is more efficient for the ALS
# algorithm, because:
# - ALS computation is already distributed over users and items
# - it is more efficient to transfer a full chunk once to a node, instead of
#   communicating to a remote node once for each step
type DistModel <: Model
    UFile::FileSpec
    PFile::FileSpec
    nfactors::Int
    lambda::Float64
    U::Nullable{ChunkedFile}
    P::Nullable{ChunkedFile}
    lambdaI::Nullable{ModelFactor}
    Pinv::Nullable{ModelFactor}
end

nusers(model::DistModel) = last(keyrange(get(model.U)))
nitems(model::DistModel) = last(keyrange(get(model.P)))
nfactors(model::DistModel) = model.nfactors

share!(model::DistModel) = nothing
localize!(model::DistModel) = nothing
function sync!(model::DistModel)
    isnull(model.U) || sync!(get(model.U))
    isnull(model.P) || sync!(get(model.P))
end

function clear(model::DistModel)
    model.U = nothing
    model.P = nothing
    model.lambdaI = nothing
    model.Pinv = nothing
end

function ensure_loaded(model::DistModel)
    model.U = read_input(model.UFile)
    model.P = read_input(model.PFile)
    model.lambdaI = model.lambda * eye(model.nfactors)
    nothing
end

# TODO
#function pinv(model::DistModel)
#end
#vec_mul_pinv(model::DistModel, v) = v * pinv(model)

function vec_mul_p(model::DistModel, v)
    res = Array(Float64, nitems(model))
    cfP = get(model.P)
    for chunk in cfP.chunks
        P = data(chunk, cfP.lrucache).val
        res[chunk.keyrange] = v * P
    end
    res
end

function prep{TI<:DistInputs}(inp::TI, nfacts::Int, lambda::Float64, model_dir::AbstractString, max_cache::Int=10)
    ensure_loaded(inp)
    t1 = time()
    @logmsg("preparing inputs...")

    nu = nusers(inp)
    ni = nitems(inp)

    UFile = DenseMatChunks(joinpath(model_dir, "U.meta"), 1, (nu, nfacts), max_cache)
    PFile = DenseMatChunks(joinpath(model_dir, "P.meta"), 2, (nfacts, ni), max_cache)
    create(UFile, zero!, min(_max_items(UFile), ceil(Int, nu/nworkers())))
    create(PFile, rand!, min(_max_items(PFile), ceil(Int, ni/nworkers())))

    cfU = read_input(UFile)
    cfP = read_input(PFile)
    for idx in 1:ni
        chunk = getchunk(cfP, idx)
        P = data(chunk, cfP.lrucache).val
        P[1,idx-first(chunk.keyrange)+1] = mean(all_user_ratings(inp, idx))
    end
    sync!(cfP)

    lambdaI = lambda * eye(nfacts)
    model = DistModel(UFile, PFile, nfacts, lambda, cfU, cfP, lambdaI, nothing)

    t2 = time()
    @logmsg("prep time: $(t2-t1)")
    model
end

function setU(model::DistModel, u::Int64, vals)
    cf = get(model.U)
    chunk = getchunk(cf, u)
    U = data(chunk, cf.lrucache).val
    U[u-first(chunk.keyrange)+1,:] = vals
    nothing
end

function setP(model::DistModel, i::Int64, vals)
    cf = get(model.P)
    chunk = getchunk(cf, i)
    P = data(chunk, cf.lrucache).val
    P[:,i-first(chunk.keyrange)+1] = vals
    nothing
end

function getU(model::DistModel, users)
    cf = get(model.U)
    Usub = Array(Float64, length(users), model.nfactors)
    for idx in 1:length(users)
        u = users[idx]
        chunk = getchunk(cf, u)
        U = data(chunk, cf.lrucache).val
        Usub[idx,:] = U[u-first(chunk.keyrange)+1,:]
    end
    Usub
end
function getP(model::DistModel, items)
    cf = get(model.P)
    Psub = Array(Float64, model.nfactors, length(items))
    for idx in 1:length(items)
        i = items[idx]
        chunk = getchunk(cf, i)
        P = data(chunk, cf.lrucache).val
        Psub[:,idx] = P[:,i-first(chunk.keyrange)+1]
    end
    Psub
end
