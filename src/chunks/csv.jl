function load{T<:Vector{UInt8}}(::Type{T}, chunk::Chunk)
    #@logmsg("loading chunk $(chunk.path)")
    open(chunk.path) do f
        seek(f, chunk.offset)
        databytes = Array(UInt8, chunk.size)
        read!(f, databytes)
        return databytes::T
    end
end

function load{T<:Matrix}(::Type{T}, chunk::Chunk)
    M = readcsv(load(Vector{UInt8}, chunk))
    M::T
end

#=
function load{T<:SparseMatrixCSC}(::Type{T}, chunk::Chunk)
    A = load(Matrix{Float64}, chunk)
    rows = convert(Vector{Int64},   A[:,1]);
    cols = convert(Vector{Int64},   A[:,2]);
    vals = convert(Vector{Float64}, A[:,3]);

    # subtract keyrange to keep sparse matrix small
    cols .-= (first(chunk.keyrange) - 1)
    sparse(rows, cols, vals)::T
end
=#
