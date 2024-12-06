function mnistconv()
    kern = (3,3)
    s = (2,2)
    θ = Chain(Conv(kern,1 => 3,relu,stride=s),
                Conv(kern,3 => 6,relu,stride=s),
                Conv(kern,6 => 9,relu,stride=s),
                Conv((2,2),9 => 12,relu))
    return Chain(θ, x->reshape(x,12,:))
end

function mnistdeconv()
    kern = (3,3)
    s = (2,2)
    ϕ = Chain(ConvTranspose((2,2),12 => 9,relu),
                   ConvTranspose((4,4),9 => 6,relu,stride=s),
                   ConvTranspose(kern,6 => 3,relu,stride=s),
                   ConvTranspose((4,4),3 => 1,relu,stride=s))
    return Chain(x->reshape(x,1,1,12,:), ϕ)
end

function mnistclassifier()
    π = Chain(Dense(3 => 5,relu),
                    Dense(5 => 10,relu),
                    softmax)
    return π
end

function mnistenc()
    mlp = Chain(Dense(12 => 6,relu),
                Dense(6 => 3,relu))

    return Chain(mnistconv(),mlp)
end

function mnistdec()
    mlp = Chain(Dense(3 => 6,relu),
                Dense(6 => 12,relu))
    return Chain(mlp, mnistdeconv())
end

function mnistloader(batchsize::Integer)
    dat = MNIST(split=:train)[:]
    target = onehotbatch(dat.targets,0:9)

    m_x,m_y,n = size(dat.features)
    X = reshape(dat.features[:, :, :], m_x, m_y, 1, n)

    loader = Flux.DataLoader((X,target),
                            batchsize=batchsize,
                            shuffle=true)
    return loader
end
