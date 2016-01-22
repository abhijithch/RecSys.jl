# Chunks are parts of a large data datastructure that can be:
#   - loaded on demand
#   - cached, but evicted on memory pressure
#   - optionally attached with locality for efficient IO (in future)
# A chunked files:
#   - has a metadata section that lists location and data range of each chunk
#   - chunk key type (data range) must have the `in`, `first`, `last`, and `length` methods defined

using LRUCache
using Base.Mmap
import Base.Mmap: sync!

# D = dimension that is split
# N = value of the other dimension, which is constant across all splits
type MemMappedMatrix{T,D,N}
    val::Matrix{T}
end
sync!(m::MemMappedMatrix) = sync!(m.val, Base.MS_SYNC | Base.MS_INVALIDATE)

type Chunk{K,V}
    path::AbstractString
    offset::Int
    size::Int
    keyrange::K
    valtype::Type{V}
    data::WeakRef
end

function Chunk(path::AbstractString, offset::Integer, size::Integer, keyrange, V)
    K = typeof(keyrange)
    Chunk{K,V}(path, offset, size, keyrange, V, WeakRef())
end

function Chunk(path::AbstractString, keyrange, V)
    size = filesize(path)
    K = typeof(keyrange)
    Chunk{K,V}(path, 0, size, keyrange, V, WeakRef())
end

function load{T,D,N}(::Type{MemMappedMatrix{T,D,N}}, chunk::Chunk)
    @logmsg("loading memory mapped chunk $(chunk.path)")
    ncells = div(chunk.size, sizeof(T))
    M = Int(ncells/N)
    dims = (D == 1) ? (M,N) : (N,M)
    A = Mmap.mmap(chunk.path, Matrix{T}, dims, chunk.offset)
    MemMappedMatrix{T,D,N}(A)
end

function load{T<:Vector{UInt8}}(::Type{T}, chunk::Chunk)
    @logmsg("loading chunk $(chunk.path)")
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
function load{T<:SparseMatrixCSC}(::Type{T}, chunk::Chunk)
    @logmsg("loading memory mapped sparse $(chunk.path)")
    open(chunk.path) do f
        seek(f, chunk.offset)
        return mmap_csc_load(f)::T
    end
end

function data{K,V}(chunk::Chunk{K,V}, lrucache::LRU)
    if chunk.data.value == nothing
        data = load(V, chunk)
        chunk.data.value = data
    end
    v = chunk.data.value
    lrucache[chunk.keyrange] = v
    v::V
end

function sync!(chunk::Chunk)
    try
        (chunk.data.value == nothing) || sync!(chunk.data.value)
    end
end

#=
function finalize_chunk_data(chunk::Chunk, data)
    #@logmsg("unloading chunk $(chunk.path)")
    # TODO: need to sync! when it gets chucked out of the lrucache
    #sync!(chunk)
    chunk.data = nothing
end
=#

type ChunkedFile{K,V}
    keyrangetype::Type{K}
    valtype::Type{V}
    metapath::AbstractString
    chunks::Vector{Chunk{K,V}}
    lrucache::LRU
end

keyrange(cf::ChunkedFile) = first(cf.chunks[1].keyrange):last(cf.chunks[end].keyrange)

sync!(cf::ChunkedFile) = empty!(cf.lrucache)

function _unload(cf, k, v)
    sync!(v)
    chunk = getchunk(cf, first(k))
    @logmsg("unloading chunk $(chunk.path)")
    chunk.data.value = nothing
    nothing
end

function ChunkedFile(metapath::AbstractString, K, V, max_cache::Integer, readonly::Bool=true)
    chunks = Chunk{K,V}[]
    meta = (filesize(metapath) == 0) ? Array(Any,0,0) : readcsv(metapath)
    for idx in 1:size(meta,1)
        fname = meta[idx,3]
        push!(chunks, Chunk(fname, 0, filesize(fname), Int(meta[idx,1]):Int(meta[idx,2]), V))
    end
    cf = ChunkedFile{K,V}(K, V, metapath, chunks, LRU{K,V}(max_cache))
    if !readonly
        cf.lrucache.cb = (k,v) -> _unload(cf, k, v)
    end
    cf
end

function writemeta(cf::ChunkedFile)
    chunkpfx = splitext(cf.metapath)[1]
    open(cf.metapath, "w") do meta
        chunks = cf.chunks
        idx = 1
        for chunk in chunks
            @logmsg("writing chunk: $(chunk.keyrange)")
            println(meta, first(chunk.keyrange), ",", last(chunk.keyrange), ",", chunkpfx, ".", idx)
            idx += 1
        end
    end
end

function getchunk{K,V}(cf::ChunkedFile{K,V}, key::Int)
    for chunk in cf.chunks
        (key in chunk.keyrange) && return chunk
    end
    error("Key not found")
end
