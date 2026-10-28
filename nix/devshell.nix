{
  lib,
  mkShell,
  stdenv,
  python,
  git,
  just,
  lychee,
  openmpi,
  uv,
  zlib,
}:

mkShell {
  nativeBuildInputs = [
    git
    just
    lychee
    # block2main, which the DMRG step runs as a subprocess, links against MPI
    openmpi
    python
    uv
  ];

  shellHook = ''
    # The python setup hook exports the site-packages of this shell's Python inputs; uv's virtual environments must
    # only see their own packages.
    unset PYTHONPATH

    # `[tool.uv] python-preference = "only-managed"` pulls prebuilt interpreters that expect a loader NixOS does not
    # provide; use this shell's CPython.
    export UV_PYTHON_PREFERENCE=only-system
    export UV_PYTHON="${python}/bin/python"

    # manylinux wheels (block2, pyscf, numpy, ...) resolve libstdc++ through the ambient loader path, which is empty
    # on NixOS.
    export LD_LIBRARY_PATH="${
      lib.makeLibraryPath [
        stdenv.cc.cc.lib
        zlib
      ]
    }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  '';
}
