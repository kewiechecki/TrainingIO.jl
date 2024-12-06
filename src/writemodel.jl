function writecsv(log::AbstractArray,path::AbstractString)
    Tables.table(log) |> CSV.write(path)
end

#function writecsv(log::AbstractArray,path::AbstractString,file::AbstractString)
#    mkpath(path)
#    Tables.table(log) |> CSV.write(path*"/"*file)
#end

function writecsv(log::Union{DataFrame,AbstractArray},path::AbstractString,file::AbstractString)
    mkpath(path)
    writecsv(log,path*"/"*file)
end

function writecsv(log::DataFrame,path::AbstractString)
    log |> CSV.write(path)
end

function writecsv(dict::Dict,path::AbstractString,file::AbstractString)
    maplab((k,X)->writecsv(X,path*k,file),dict)
end 

function writecsv(dict::Dict,path::AbstractString)
    maplab((k,X)->writecsv(X,path,k*".csv"),dict)
end 

function savemodel(M,path::AbstractString)::Nothing
    M = cpu(M)
    state = Flux.state(M) |> cpu;
    #jldsave(path*".jld2";state)
    @save path*".jld2" M state
end

function savemodel(M,path::AbstractString,file::AbstractString)::Nothing
    mkpath(path)
    M = cpu(M);
    state = Flux.state(M) |> cpu;
    #jldsave(path*"/"*file*".jld2";state,arch)
    @save path*"/"*file*".jld2" M state
end

function savemodel(M,log,path,file)::Nothing
    mkpath(path)
    state = Flux.state(M) |> cpu;
    arch = typeof(M)
    jldsave(path*"/"*file*".jld2";state,arch)
    Tables.table(log) |> CSV.write(path*"/"*file*"_loss.csv")
end

