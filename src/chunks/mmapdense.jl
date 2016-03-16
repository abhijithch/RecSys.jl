# D = dimension that is split
# N = value of the other dimension, which is constant across all splits
type MemMappedMatrix{T,D,N}
    val::Matrix{T}
end
sync!(m::MemMappedMatrix) = sync!(m.val, Base.MS_SYNC | Base.MS_INVALIDATE)

function load{T,D,N}(::Type{MemMappedMatrix{T,D,N}}, chunk::Chunk)
    #@logmsg("loading memory mapped chunk $(chunk.path)")
    ncells = div(chunk.size, sizeof(T))
    M = Int(ncells/N)
    dims = (D == 1) ? (M,N) : (N,M)
    A = Mmap.mmap(chunk.path, Matrix{T}, dims, chunk.offset)
    MemMappedMatrix{T,D,N}(A)
end
