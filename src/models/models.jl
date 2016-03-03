include("als_model.jl")
include("als_dist_model.jl")

setU{M<:Model}(model::M, u::Int64, vals) = setU(model.U, u, vals)
setU(NU::Nullable, u::Int64, vals) = setU(get(NU), u, vals)
function setU{M<:ModelFactor}(U::M, u::Int64, vals)
    U[u,:] = vals
    nothing
end

setP{M<:Model}(model::M, i::Int64, vals) = setP(model.P, i, vals)
setP(NP::Nullable, i, vals) = setP(get(NP), i, vals)
function setP{M<:ModelFactor}(P::M, i::Int64, vals)
    P[:,i] = vals
    nothing
end

getU{M<:Model}(model::M, users) = getU(model.U, users)
getU(NU::Nullable, users) = getU(get(NU), users)
getU{M<:ModelFactor}(U::M, users) = U[users, :]

getP{M<:Model}(model::M, items) = getP(model.P, items)
getP(NP::Nullable, items) = getP(get(NP), items)
getP{M<:ModelFactor}(P::M, items) = P[:, items]

vec_mul_p{M<:Model}(model::M, v) = vec_mul_p(model.P, v)
vec_mul_p(NM::Nullable, v) = vec_mul_p(get(NM), v)
vec_mul_p{M<:ModelFactor}(model::M, v) = v * model
