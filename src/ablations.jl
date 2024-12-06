@doc raw"""
`maskrow(E::AbstractArray,i::Integer,invert=false) -> CuArray`

Workaround for ablating rows of `CuArray`s.
"""
function maskrow(E::AbstractArray,i::Integer,invert=false)
    m,n = size(E)
    mask = sparse(rep(i,n),1:n,1,m,n)
    if(invert)
        mask = 1 .- mask
    end
    return mask |> gpu
end

@doc raw"""
`zeroabl(E::CuArray,i::Integer,invert=false) -> CuArray`

Workaround for lack of support for indexing `CuArray`s. Returns a copy of `E` with `E[i,:]` set to 0. If `invert=true`, instead returns a copy with all values `E[Not(i),:]` set to 0.
"""
function zeroabl(E::AbstractArray,i::Integer,invert=false)
    return E .* maskrow(E,i,invert)
end

@doc raw"""
`meanabl(E::CuArray,i::Integer,invert=false -> CuArray`

Workaround for lack of support for indexing `CuArray`s. Returns a copy of `E` with `E[i,:]` set to `mean(E[i,:])`. If `invert=true`, instead returns a copy with all `E[j,:]` for `j != i` to `mean(E[j,:]`.
"""
function meanabl(E::AbstractArray,i::Integer,invert=false)
    μ = mean(E,dims=2)
    mask = maskrow(E,i,invert)
    x = E .* mask
    y = μ .* (1 .- mask)
    return x .+ y
end
