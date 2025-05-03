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
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils , DictMap }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = system == "x86_64-linux";
          };
        };

        juliaPkgs = pkgs.juliaPackages;
        shellPkgsNested = with pkgs; [ # Keep shell packages definition separate
          julia git stdenv.cc gfortran stdenv.cc.cc.lib
          (lib.optional stdenv.isLinux cudaPackages.cudatoolkit)
          (lib.optional stdenv.isLinux cudaPackages.cudnn)
        ];

        # --- Get the actual DictMap Nix package derivation ---
        # Assumes the DictMap flake exports packages.default correctly
        dictMapPkg = DictMap.packages.${system}.default;
        shellPkgs = pkgs.lib.flatten shellPkgsNested;

        # Build TrainingIO.jl
		/*
        trainingIObuilt = juliaPkgs.buildJuliaPackage {
          pname = "TrainingIO";
          version = "0.1.1"; # TODO: FIX THIS
          src = ./.;

          # Propagate runtime libs (gfortran) AND the dependency package (dictMapPkg)
          # This signals that users/builders of TrainingIO need these.
          # The Nix Julia hooks MIGHT use this to make dictMapPkg available
          # to the Julia build environment for TrainingIO.
          propagatedBuildInputs = [
            pkgs.gfortran
            dictMapPkg  # <<< Pass the actual package derivation here
          ];
        };
		  */

      in {
        #packages.trainingIO = trainingIObuilt;
        #packages.default = self.packages.${system}.trainingIO;

        devShell = pkgs.mkShell { # Use pkgs. explicitly
          name = "trainingio-dev-shell";
          # buildInputs only needs the runtime stuff for the shell itself
          buildInputs = shellPkgs;

          shellHook = ''
            source ${pkgs.git}/share/bash-completion/completions/git-prompt.sh
            export JULIA_PROJECT="@."
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath shellPkgs}";

            echo "Nix dev shell for TrainingIO.jl activated." # Corrected package name
            echo "Julia environment uses Project.toml (JULIA_PROJECT=@.)."
            # Debug check
            echo "--- Checking LD_LIBRARY_PATH ($LD_LIBRARY_PATH) for libquadmath.so.0 ---"
            ( IFS=: ; for p in $LD_LIBRARY_PATH; do if [ -f "$p/libquadmath.so.0" ]; then echo "  FOUND in $p"; fi; done )
            echo "----------------------------------------------------"
          '';
        };
      }
    );
}
