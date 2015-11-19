zero(::Type{AbstractString}) = ""

function save(model, filename::AbstractString)
    open(filename, "w") do f
        serialize(f, model)
    end
    nothing
end

function load(filename::AbstractString)
    open(filename, "r") do f
        deserialize(f)
    end
end

type DlmFile <: FileSpec
    name::AbstractString
    dlm::Char
    header::Bool

    function DlmFile(name::AbstractString, dlm::Char=',', header::Bool=false)
        new(name, dlm, header)
    end
end

function read_input(fspec::DlmFile)
    # read file and skip the header
    F = readdlm(fspec.name, fspec.dlm, header=fspec.header)
    fspec.header ? F[1] : F
end
