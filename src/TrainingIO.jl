module TrainingIO

using Reexport
@reexport using Flux, ProgressMeter, JLD2, CSV, DataFrames
using Dates, Plots, Tables
@reexport using DictMap

using MLDatasets, OneHotArrays
using StatsBase, InvertedIndices

export zeroabl,meanabl
export sampledat,scaledat,unhot
export mnistenc, mnistdec, mnistclassifier, mnistloader

export update!,train!,load!,savemodel,loadmodel,readloss
export writecsv, readcsv, readmat
export date,savepath,savedate

include("savedate.jl")
include("writemodel.jl")
include("readmodel.jl")

@doc raw"""
`update!(M, loss, opt::Flux.Optimiser) -> typeof(loss(M,x,y))`

`update!(M, x, y, loss, opt::Flux.Optimiser) -> typeof(loss(M,x,y))`

Updates the parameters of `M` using `Flux.update!`.

If `x` and `y` are provided, `loss` is interpreted as a trinary function

`loss::((typeof(x) -> typeof(y)) -> typeof(x) -> typeof(y) -> Float`.

Otherwise, `loss` is interpreted as a unary function to be passed to `Flux.withgradient`.

This is a lower-level function for defining training loops. For a plug-and-play training loop, use `train!` 

See also: `train!`, Optimisers.update!`, `Optimisers.setup`, `Flux.withgradient`.
"""
function update!(M,x,y,
                 loss,
                 opt) #Flux.Optimiser removed as type in Flux 16.0
    #If x,y are specified, use them to construct a curried loss function.

    #x = gpu(x)
    #y = gpu(y)
    f = m->loss(m(x),y)
    update!(M,f,opt)
    #state = Flux.setup(opt,M)
    #l,∇ = Flux.withgradient(f,M)
    #Flux.update!(state,M,∇[1])
    #return l
end

function update!(M,
                 loss,
                 opt)
    state = Flux.setup(opt,M)
    l,∇ = Flux.withgradient(loss,M)
    #(print∘sum)(∇[1].weight)
    Flux.update!(state,M,∇[1])
    return l
end

@doc raw"""
`train!(M,path,
        loss,
        loader::Flux.DataLoader,
        opt::Flux.Optimiser,
        epochs::Integer; ignoreY=true, savecheckpts=true) -> Matrix`

Trains a model `M` using data in `loader`, cycling through `loader` `epochs` times. Saves the final result to `path*"/final.jld2`. Loss for each update step is saved in `path*"/loss.csv"`. Returns a `Matrix` with the results of `loss` for each epoch.

This is a high level function intended to streamline I/O. If a custom training loop is desired, use `update!`.

If `loss` returns multiple arguments: only the first is used for optimisation, but all will be written to `loss.csv`.

If `savecheckpts`, a copy of the model is saved after each epoch.

`loss` should accept a model, model input, and expected model output. Standard loss functions such as `Flux.mse` should be wrapped as `(M,x,y)->Flux.mse(M(x),y)`. This can be accomplished with `Autoencoders.loss`.

See also: `Autoencoders.loss` `update!`, `Optimisers.update!`, `Flux.trainmodel!`
"""
function train!(M,
                loader::Flux.DataLoader,
                opt,
                epochs::Integer,
                loss,
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
                opt,
                epochs::Integer,
                loss;
                kwargs...)
    log = []
    train!(M,loader,opt,epochs,loss,log;kwargs...)
    return log
end

function train!(M,path::String,
                loss,
                loader::Flux.DataLoader,
                opt,
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
            if ignoreY
                y = x
            end
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

include("ablations.jl")
include("preprocessing.jl")
include("mnist.jl")

end # module TrainingIO
