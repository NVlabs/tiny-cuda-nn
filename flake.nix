{
  description = "tiny-cuda-nn dev env";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            gcc
            gdb
            cmake
            pkg-config
            binutils
            zlib

            python3
            stdenv.cc.cc.lib

            cudatoolkit
            cudaPackages.cuda_cudart
            cudaPackages.cuda_nvrtc
            cudaPackages.cuda_nvtx
          ];

          shellHook = ''
            export CUDA_PATH="${pkgs.cudatoolkit}"
            export CLANGD_CUDA_INCLUDE="${pkgs.cudatoolkit}"

            # Required for Python libs like numpy to find dynamic libs
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:''${LD_LIBRARY_PATH:-}"
          '';
        };
      }
    );
}
