# Diff Physics
this project aims to be a differentiable simulation editing tool.
## Current State
- a differentiable PD solver without contact
  - [reference](https://arxiv.org/abs/2106.05306)
- a differentiable cloth simulation example
## Run This
- clone and build [my forked version of taichi-lang](https://github.com/windwhiterain/taichi) on branch "local"
- this project depends on the forked taichi by relative path reference (check or modify this in [pyproject](pyproject.toml)), you can put them in the same directory.
- run [example](tests/__init__.py) 
## Future Work
- add contact
- replace taichi-lang with another backend if there is a better one 