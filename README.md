# Thesis Code Repository
This repository is comprised of 10 jupyter notebooks, tables, figures, shift data and resuults data, all used for my thesis on state-dependent modifications
of the Quasi-Oppositional Social Optimal Foraging Algorithm (QOS-OFA).

The thesis utilizes 3 algorithms, the original QOS-OFA and two proposed ones, IT-QOS-OFA and IT-SIA-QOS-OFA. More information about the algorithms is provided in the thesis. 
The structure of the project is as follows:

## Main files

### `Thesis_QOS-OFA.ipynb`

Applies QOS-OFA algorithm on both 2D and 30D problems, following the CEC- benchmark suite.

### `Individual_temp.ipynb`

This notebook introduces the first new proposition which is individual temeprature, the notebook contains code for computing performance, plots  and fine tunning paramater sections.

### `Individual_temp_Spread_accept.ipynb`

This is a second notebook which has a new proposition within it, this one also contains plots and paramater tunning, as well as code for computing results.

### `CEC_convergance_plotting.ipynb`

This notebook takes results from previous stated notebooks, and builds result plots.

### `Statistical_and_Numerical_Analysis.ipynb`

This notebook is the heart of the results section, it contains all major calculations which are stated within the thesis.

### `Thesis_QOS-OFA_PLOTTING.ipynb`

Thi notebook is used for easier plots which are not heavly used for results.

## Python files

### `benchmark_adapter.py`

Helper code for connecting the benchmark functions to the algorithm notebooks.

### `benchmarks_2D.py`

Two-dimensional benchmark functions used for visual explanation and diagnostic plots.

### `benchmarks_CEC.py`

CEC benchmark interface are used for the main experimental evaluation.

## Folders

### `Figures`

Contains the generated thesis figures.

### `Tables`

Contains exported tables used in the thesis.

### `Excel_sheets_varius_results`

Contains Excel results used for statistican and numerical analysis.


### `Shift_data`

Contains additional data required for some benchmark functions, due to the usage of CEC 2020 benchmark suite

## Environment

The environment can be created using the provided Conda YAML file: conda env create -f thesis_cpu.yaml, which then needs to also be activated.


