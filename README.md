# RxInfer experiments and comparisons

This respository contains experimental benchmarks across three packages: [RxInfer.jl](https://github.com/biaslab/RxInfer.jl), 
[ForneyLab.jl](https://github.com/biaslab/ForneyLab.jl) and [Turing.jl](https://github.com/TuringLang/Turing.jl).
The codebase uses the Julia programming language of version 1.9.0. 
To ensure reproducibility the experimanets have been prepared with the help of the 
[DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl) package. 

The repository has been authored by Bagaev Dmitry: d.v.bagaev@tue.nl

Folder structure:

- `research`: Jupyter notebooks, which were used during the research phase. These notebook could be broken, not working properly or do not show any results. This folder is used primarily for research purposes.
- `notebooks`: Jupyter notebooks with experiments, which were used in the thesis. See `notebooks/README.md` for more info.
- `src`: shared code for experiments. See `src/README.md` for more info.
- `data`: (optional) does not exist by default, caches benchmark data.
- `plots`: (optional) does not exist by default, caches plots data.

# TODOs:
- add `src/README.md`
- add `notebooks/README.md`

