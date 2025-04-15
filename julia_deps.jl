using Pkg
Pkg.instantiate()

Pkg.add("cuDNN")
Pkg.add("StructArrays")

for (pkg, path) in [
    ("DictMap", "/nix/store/bsaib7phfah203wa7h24097fn18hjfwg-source"),
]
    try
        @eval import \$(Symbol(pkg))
        println("Package ", pkg, " is already installed.")
    catch e
        println("Developing package ", pkg, " from ", path)
        try
            Pkg.develop(path=path)
            #Pkg.precompile(only=[pkg])
        catch e
            println("Error precompiling ", pkg, ": ", e)
            #exit(1)
        end
    end
end

Pkg.update()
using TrainingIO

