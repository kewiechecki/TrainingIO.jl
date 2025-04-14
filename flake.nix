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
    TrainingIO = {
      url = "github:kewiechecki/TrainingIO.jl";
      flake = false;
    };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { 
          inherit system;
          config.allowUnfree = true;
          # config.cudaSupport = system == "x86_64-linux";
		};

        # Get library paths from the stdenv compiler and from gfortran.
        # gccPath = toString pkgs.stdenv.cc.cc.lib;
        # gfortranPath = toString pkgs.gfortran;

        # Define the multi-line Julia script.
        # NOTE: The closing delimiter (two single quotes) MUST be flush with the left margin.
        juliaScript = ''
using Pkg
Pkg.instantiate()

for (pkg, path) in [
    ("REPLVim", "__REPLVim__"),
    ("TrainingIO", "__TRAININGIO__"),
    ("DictMap", "__DICTMAP__"),
]
    try
        @eval import __DOLLAR_PLACEHOLDER__(Symbol(pkg))
        println("Package ", pkg, " is already installed.")
    catch e
        println("Developing package ", pkg, " from ", path)
        try
            Pkg.develop(path=path)
            Pkg.precompile(only=[pkg])
        catch e
            println("Error precompiling ", pkg, ": ", e)
            #exit(1)
        end
    end
end

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
		  ];
          shellHook = ''
source ${git}/share/bash-completion/completions/git-prompt.sh

cat > julia_deps.jl <<'EOF'
${juliaScript}
EOF

# Activate the project and instantiate dependencies.
julia --project=. julia_deps.jl
          '';
        };
      }
    );
}

