
# ∀ A:Type m,n:Int -> [A m n] -> k:Int -> ([A m k],[A m n-k])
function sampledat(X::AbstractArray,k)
    _,n = size(X)
    sel = sample(1:n,k,replace=false)
    test = X[:,sel]
    train = X[:,Not(sel)]
    return test,train
end

# ∀ m,n:Int -> [Float m n] -> [Float m n]
#scales each column (default) or row to [-1,1]
function scaledat(X::AbstractArray,dims=1)
    Y = X ./ maximum(abs.(X),dims=dims)
    Y[isnan.(Y)] .= 0
    return Y
end

function unhot(x)
    map(i->i[1],argmax(x,dims=1)) .- 1
end

