function loadmodel(path::AbstractString,file="final.jld2")
    @load path*"/"*file M state
    Flux.loadmodel!(M, state)
    return M
end

function readloss(path::AbstractString,file="loss.csv")::DataFrame
    return (DataFrame ∘ CSV.File)(path*"/"*file)
end

function load!(M,path::AbstractString,file="final.jld2")
    #state = JLD2.load(path*"/"*file,"state")
    @load path*"/"*file M state
    L = CSV.File(path*"/loss.csv").Column1
    Flux.loadmodel!(M,state)
    return M,L,path
end

function load!(f::Function,path::AbstractString,file="final.jld2")
    M = f() |> gpu
    return load!(M,path)
end

function readcsv(file)::DataFrame
    CSV.File(file) |> DataFrame
end

function readmat(file)::Matrix
    readcsv(file) |> Matrix
end
