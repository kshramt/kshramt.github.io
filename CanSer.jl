"""
Supported data types:

- Integer (64 bits)
- Float (64 bits)
- String (UTF8)
- Vector
- Dictionary

Little endiannness is assumed.
"""
module CanSer

save(io::IO, x::Float64) = write(io, 'f', x)
save(io::IO, x::Int64) = write(io, 'i', x)
save(io::IO, x::String) = write(io, 's', sizeof(x), x)

function save(io::IO, x::Vector)
    write(io, 'l', length(x))
    for v in x
        save(io, v)
    end
end

function save(io::IO, x::Dict)
    write(io, 'd', length(x))
    for k in sort(collect(keys(x)))
        save(io, k)
        save(io, x[k])
    end
end


function load(io::IO)
    tag = read(io, Char)
    if tag == 'f'
        read(io, Float64)
    elseif tag == 'i'
        read(io, Int64)
    elseif tag == 's'
        String(read(io, read(io, Int64)))
    elseif tag == 'l'
        [load(io) for _ in 1:read(io, Int64)]
    elseif tag == 'd'
        Dict((k = load(io); v = load(io); k=>v) for _ in 1:read(io, Int64))
    else
        throw(ArgumentError("Invalid tag $tag for $io"))
    end
end

end
