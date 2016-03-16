# In distributed memory mode, the model may be too large to load into memory on
# a single node. It is therefore split by one of its dimensions into blobs.
#
# All matrix operations are done on the node where required, just that not all
# blobs are loaded at once at any time. This is more efficient for the ALS
# algorithm, because:
# - ALS computation is already distributed over users and items
# - it is more efficient to transfer a full blob once to a node, instead of
#   communicating to a remote node once for each step
type DistModel <: Model
    UFile::FileSpec
    PFile::FileSpec
    nfactors::Int
    lambda::Float64
    U::Nullable{ModelFactor}
    P::Nullable{ModelFactor}
    lambdaI::Nullable{ModelFactor}
    Pinv::Nullable{ModelFactor}
end

nusers(model::DistModel) = size(get(model.U))[2]
nitems(model::DistModel) = size(get(model.P))[2]
nfactors(model::DistModel) = model.nfactors

share!(model::DistModel) = nothing
localize!(model::DistModel) = nothing

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
#mul_pinv(model::DistModel, v) = v * pinv(model)

function prep{TI<:DistInputs}(inp::TI, nfacts::Int, lambda::Float64, model_dir::AbstractString, max_cache::Int=10)
    ensure_loaded(inp)
    t1 = time()
    @logmsg("preparing inputs...")

    nu = nusers(inp)
    ni = nitems(inp)

    Udir = DenseBlobs(joinpath(model_dir, "U"))
    Pdir = DenseBlobs(joinpath(model_dir, "P"))
    isdir(model_dir) || mkdir(model_dir)
    isdir(Udir.name) || mkdir(Udir.name)
    isdir(Pdir.name) || mkdir(Pdir.name)

    Usz = (nfacts, nu)
    Psz = (nfacts, ni)
    U = create(Udir, Float64, Usz, zeros, min(_max_items(Float64,Usz), ceil(Int, nu/nworkers())))
    P = create(Pdir, Float64, Psz, rand, min(_max_items(Float64,Psz), ceil(Int, ni/nworkers())))

    for idx in 1:ni
        P[1,idx] = mean(all_user_ratings(inp, idx))
    end
    save(U)
    save(P)

    lambdaI = lambda * eye(nfacts)
    model = DistModel(Udir, Pdir, nfacts, lambda, U, P, lambdaI, nothing)

    t2 = time()
    @logmsg("prep time: $(t2-t1)")
    model
end
