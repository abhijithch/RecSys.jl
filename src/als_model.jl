type Model
    U::ModelFactor
    P::ModelFactor
    nfactors::Int
    lambda::Float64
    lambdaI::Nullable{ModelFactor}
    Pinv::Nullable{ModelFactor}
end

nusers(model::Model) = size(model.U, 1)
nitems(model::Model) = size(model.P, 2)
nfactors(model::Model) = model.nfactors

function share!(model::Model)
    isa(model.U, SharedArray) || (model.U = share(model.U))
    isa(model.P, SharedArray) || (model.P = share(model.P))

    lambdaI = get(model.lambdaI)
    isa(lambdaI, SharedArray) || (model.lambdaI = share(lambdaI))

    if !isnull(model.Pinv)
        Pinv = get(model.Pinv)
        isa(Pinv, SharedArray) || (model.Pinv = share(Pinv))
    end
    nothing
end

function localize!(model::Model)
    isa(model.U, SharedArray) && (model.U = copy(model.U))
    isa(model.P, SharedArray) && (model.P = copy(model.P))
    nothing
end

function clear(model::Model)
    model.lambdaI = nothing
    model.Pinv = nothing
end

function pinv(model::Model)
    if isnull(model.Pinv)
        # since: I = (P * Pt) * inv(P * Pt)
        # Pinv = Pt * inv(P * Pt)
        P = model.P
        PT = P'
        model.Pinv = PT * inv(P * PT)
    end
    get(model.Pinv)
end

vec_mul_p(model::Model, v) = v * model.P
vec_mul_pinv(model::Model, v) = v * pinv(model)

function prep(inp::Inputs, nfacts::Int, lambda::Float64)
    ensure_loaded(inp)
    t1 = time()
    logmsg("preparing inputs...")

    nu = nusers(inp)
    ni = nitems(inp)

    U = zeros(nu, nfacts)
    P = rand(nfacts, ni)
    for idx in 1:ni
        P[1,idx] = mean(all_user_ratings(inp, idx))
    end

    lambdaI = lambda * eye(nfacts)
    model = Model(U, P, nfacts, lambda, lambdaI, nothing)

    t2 = time()
    logmsg("prep time: $(t2-t1)")
    model
end

@inline function setU(model::Model, u::Int64, vals)
    model.U[u,:] = vals
    nothing
end

@inline function setP(model::Model, i::Int64, vals)
    model.P[:,i] = vals
    nothing
end

getU(model::Model, users) = model.U[users, :]
getP(model::Model, items) = model.P[:, items]
