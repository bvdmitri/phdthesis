# RxInfer experiments and comparisons

This respository contains experimental benchmarks across three packages: [RxInfer.jl](https://github.com/biaslab/RxInfer.jl), 
[ForneyLab.jl](https://github.com/biaslab/ForneyLab.jl) and [Turing.jl](https://github.com/TuringLang/Turing.jl).
The codebase uses the Julia programming language of version 1.9.0. 
To ensure reproducibility the experimanets have been prepared with the help of the 
[DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl) package. 

The repository has been authored by Bagaev Dmitry: d.v.bagaev@tue.nl

Folder structure:

- `research`: Jupyter notebooks, which were used during the research phase. These notebook could be broken and may not work properly. This folder is used primarily for research/playground purposes.
- `notebooks`: Jupyter notebooks with experiments, which were used in the thesis. See `notebooks/README.md` for more info.
- `src`: shared code for experiments.
- `data`: (optional) does not exist by default, caches benchmark data. Can be downloaded from the releases section of the repository.
- `plots`: (optional) does not exist by default, caches plots data.

All important notebooks with code examples and benchmark experiments are present in `notebooks/` folder. Notebooks have been written with the [Jupyter](https://jupyter.org/). By default `IJulia` is not included in `Project.toml`. To run and explore experiments you will need Jupyter and IJulia kernel installed on your system.

## Running experiments

To (locally) reproduce this project, do the following:

0. Install the [Julia](https://julialang.org/) programming language of 1.9.x version, [Jupyter](https://jupyter.org/) notebooks and [`IJulia`](https://github.com/JuliaLang/IJulia.jl) kernel.

   **Important**: Experiments have been verified using Julia of version 1.9.0. Experiments should run on other versions of Julia, but benchmark data might be different for different versions of Julia. To manage different version of Julia we can recommended [juliup](https://github.com/JuliaLang/juliaup) - cross-platform Julia version manager.

1. Install `DrWatson` julia package with the following command:

   ```bash
   julia -e 'import Pkg; Pkg.add("DrWatson")'
   ```
2. Download this code base. Notice that raw data is typically not included in the
   git history and may need to be downloaded independently (see Step 4.).
3. Run the following command in the root directory of this repository:
   
   ```bash
   julia --project -e 'import Pkg; Pkg.instantiate()'
   ```
   This command will replicate the exact same environment as was used during the experiments. Any other `Pkg` related commands may alter this enviroment or/and change versions of packages.
4. (Optional) Download precomputed benchmark .JLD2 files from the releases page and unzip them in `data` folder.
   
   Precomputed benchmarks drastically reduce the amount of time needed to run notebooks and represent the exact same data used in the thesis. Note, however, that benchmark results on different machines may differ. Always compare the `versioninfo()` output in the notebooks.

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.
