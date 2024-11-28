module TrainingIO
export update!,train!,load!,writecsv,savemodel,loadmodel,readloss
export date,savepath,savedate

using Flux,ProgressMeter, JLD2,Tables,CSV,DataFrames
using Dates,Plots
using DictMap

include("savedate.jl")
include("writemodel.jl")
include("readmodel.jl")

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
                ignoreY=false,
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

end # module TrainingIO
