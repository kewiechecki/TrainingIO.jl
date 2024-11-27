module TrainingIO
export update!,train!,load!,writecsv,savemodel,loadmodel,readloss

using Flux,ProgressMeter, JLD2,Tables,CSV,DataFrames

function update!(M,x,y,
                 loss::Function,
                 opt::Flux.Optimiser)
    #x = gpu(x)
    #y = gpu(y)
    f = m->loss(m(x),y)
    state = Flux.setup(opt,M)
    l,∇ = Flux.withgradient(f,M)
    Flux.update!(state,M,∇[1])
    return l
end

function update!(M,
                 loss::Function,
                 opt::Flux.Optimiser)
    state = Flux.setup(opt,M)
    l,∇ = Flux.withgradient(loss,M)
    #(print∘sum)(∇[1].weight)
    Flux.update!(state,M,∇[1])
    return l
end

function writecsv(log::AbstractArray,path::AbstractString)
    Tables.table(log) |> CSV.write(path)
end

function writecsv(log::AbstractArray,path::AbstractString,file::AbstractString)
    mkpath(path)
    Tables.table(log) |> CSV.write(path*"/"*file)
end

function writecsv(log::DataFrame,path::AbstractString,file::AbstractString)
    mkpath(path)
    log |> CSV.write(path*"/"*file)
end

function writecsv(log::DataFrame,path::AbstractString)
    mkpath(path)
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

function train!(M,
                loader::Flux.DataLoader,
                opt::Flux.Optimiser,
                epochs::Integer,
                loss::Function,
                log;
                prefn = identity,
                postfn=identity,
                ignoreY=false,
                savecheckpts=false,
                path="")
    if length(path) > 0
        mkpath(path)
    end
    #f = (E,y)->loss(postfn(E),y)
    @showprogress map(1:epochs) do i
        map(loader) do (x,y)
            x = gpu(x)
            y = gpu(y)
            if ignoreY
                y = x
            end
            #E = prefn(x)
            #l = update!(M,E,y,f,opt)
            f = M->loss((postfn ∘ M ∘ prefn)(x),y)
            l = update!(M,f,opt)
            push!(log,l)
        end
        if savecheckpts
            savemodel(M,path*"/"*string(i))
        end
    end
    if length(path) > 0
        savemodel(M,path*"/final")
        Tables.table(log) |> CSV.write(path*"/loss.csv")
        plotloss(log,"loss",path*"/loss.pdf")
    end
end

function train!(M,
                loader::Flux.DataLoader,
                opt::Flux.Optimiser,
                epochs::Integer,
                loss::Function;
                kwargs...)
    log = []
    train!(M,loader,opt,epochs,loss,log;kwargs...)
    return log
end

function train!(M,path::String,
                loss::Function,
                loader::Flux.DataLoader,
                opt::Flux.Optimiser,
                epochs::Integer;
                savecheckpts=true)
    if length(path) > 0
        mkpath(path)
    end
    log = []
    @showprogress map(1:epochs) do i
        map(loader) do (x,y)
            x = gpu(x)
            y = gpu(y)
            f = m->loss(m,x,y)
            l = update!(M,f,opt)
            push!(log,[l...]')
        end
        if savecheckpts
            savemodel(M,path*"/"*string(i))
        end
    end
    log = vcat(log...)
    if length(path) > 0
        savemodel(M,path*"/final")
        #Tables.table(vcat(log...)) |> CSV.write(path*"/loss.csv")
        writecsv(log,path*"/loss.csv")
#        plotloss(log[:,1],"loss",path*"/loss.pdf")
    end
    return log
end

function loadmodel(path::AbstractString,file="final.jld2")
    @load path*"/"*file M state
    Flux.loadmodel!(M, state)
    return M
end

function readloss(path::AbstractString,file="loss.csv")
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

end # module TrainingIO
