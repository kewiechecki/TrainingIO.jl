{
  description = "Flake for TrainingIO.jl";
  nixConfig = {
    bash-prompt = "\[TrainingIO$(__git_ps1 \" (%s)\")\]$ ";
  };

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    DictMap = {
      url = "github:kewiechecki/DictMap.jl";
      flake = true;
    };
  };

  outputs = { self, nixpkgs, flake-utils , DictMap }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { 
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = system == "x86_64-linux";
		};

        # Get library paths from the stdenv compiler and from gfortran.
        gccPath = toString pkgs.stdenv.cc.cc.lib;
        gfortranPath = toString pkgs.gfortran;

        # Define the multi-line Julia script.
        # NOTE: The closing delimiter (two single quotes) MUST be flush with the left margin.
        juliaScript = ''
using Pkg
Pkg.instantiate()

Pkg.add("cuDNN")
Pkg.add("StructArrays")

for (pkg, path) in [
    ("DictMap", "__DICTMAP__"),
]
    try
        @eval import __DOLLAR_PLACEHOLDER__(Symbol(pkg))
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
'';

		
      in {
        # A derivation for your package.
        packages.autoencoders = pkgs.stdenv.mkDerivation {
          name = "TrainingIO.jl";
          src = ./.;
          # If your package is purely interpreted, no build phase is needed.
          # You can extend this if you have precompilation or other build steps.
        };

        # A development shell that provides Julia with your package instantiated.
        devShell = with pkgs; mkShell {
          name = "trainingio-dev-shell";
          buildInputs = [ 
		    julia 
			git
			stdenv.cc
			gfortran
		  ];
          shellHook = ''
source ${git}/share/bash-completion/completions/git-prompt.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${gfortranPath}/lib:${gccPath}/lib:${gccPath}/lib64
echo $LD_LIBRARY_PATH

cat > julia_deps.jl <<'EOF'
${juliaScript}
EOF

# Replace placeholders with actual paths.
sed -i 's|__DICTMAP__|${toString DictMap}|g' julia_deps.jl

# Replace the dollar placeholder with a literal dollar sign.
sed -i 's|__DOLLAR_PLACEHOLDER__|\\$|g' julia_deps.jl

# Activate the project and instantiate dependencies.
julia --project=. julia_deps.jl
'';
        };
      }
    );
}

