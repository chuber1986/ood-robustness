# OOD-Robustness

<!-- [![pypi](https://img.shields.io/pypi/v/ood-robustness.svg)](https://pypi.org/project/ood-robustness/) -->
<!-- [![python](https://img.shields.io/pypi/pyversions/ood-robustness.svg)](https://pypi.org/project/ood-robustness/) -->
<!-- [![Build Status](https://github.com/chuber1986/ood-robustness/actions/workflows/dev.yml/badge.svg)](https://github.com/chuber1986/ood-robustness/actions/workflows/dev.yml) -->
<!-- [![codecov](https://codecov.io/gh/chuber1986/ood-robustness/branch/main/graphs/badge.svg)](https://codecov.io/git/chuber1986/ood-robustness) -->
<!--  -->

<!-- [Project](https://sites.google.com/) **|** [Paper](https://aip.scitation.org/doi/full/10.1063/5.0020404/) -->

[Christian Huber](https://www.researchgate.net/profile/Christian-Huber-21), 
[Bernhard Lehner](https://www.researchgate.net/profile/Bernhard-Lehner-2), 
[Claus Hofmann](https://www.claus-hofmann.com/), 
[Wei Lin](https://wlin-at.github.io/),
[Bernhard Moser](https://www.researchgate.net/profile/Bernhard-Moser-4), 
[Sepp Hochreiter](https://www.researchgate.net/profile/Sepp-Hochreiter)

# Overlooked Aspects in the Evaluation of Out-Of-Distribution Detection Methods, NeuRIPS, 2024 (review)

-   Documentation: <https://github.com/chuber1986/ood-robustness.git/>
-   GitHub: <https://github.com/chuber1986/ood-robustness.git/>
-   Free software: MIT license



## Abstract
Out-of-distribution (OOD) detection identifies samples outside the data distribution used to train a machine learning model.    
In safety-critical domains like autonomous driving, OOD detection is crucial for enhancing the trustworthiness and reliability of machine learning systems.
Despite significant advancements in analyzing and enhancing the robustness of neural networks against various corruptions in image understanding, the degree to which this translates into the robustness of OOD detectors remains largely unexplored.

In this paper, we make the first attempt to incorporate robustness-related results into OOD detection benchmarks.
Our experimental setup includes benchmarking various types of global and local image interventions, such as color manipulations, blurring, and single-pixel changes.
Additionally, we highlight the dangers and limitations of using established image datasets, where the presence and severity of preprocessing artifacts, like compression or resizing, are often unknown but can significantly influence results.
To address these limitations, we introduce Shapetastic, a dataset framework designed to generate images with corresponding ground truth annotations enabling previously impossible benchmarks.
The results of our extensive experiments indicate that all state-of-the-art OOD detectors exhibit inherent and sometimes counterintuitive sensitivities that have not yet been addressed in the research literature.
Our codes and the novel synthetic dataset (Shapetastic OOD) generated with Shapetastic are available on [https://github.com/chuber1986/ood-robustness.git](https://github.com/chuber1986/ood-robustness.git)

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements and Dependencies](#requirements-and-dependencies)
3. [Getting started](#getting-started)
4. [Usage](#usage)

## Introduction

<a name="introduction"></a>
This repository containes the nessesary code to reproduce the results of the "Overlooked Aspects in the Evaluation of Out-Of-Distribution Detection Methods" paper.

## Requirements and Dependencies

<a name="requirements-and-dependencies"></a>

-   Tested on Linux
-   Python (tested with Python3.11)
-   [Hyperpyper](https://github.com/berni-lehner/hyperpyper)
-   [Shapetastic](https://github.com/berni-lehner/shapetastic)
-   [Hopfield-Boosting](https://github.com/ml-jku/hopfield-boosting)

## Getting started

<a name="getting-started"></a>
Download repository:

```bash
$ git clone https://github.com/chuber1986/ood-robustness.git
$ cd ood-robustness
```

It's recommended to use the SSH link, starting with "git://" instead of hte HTTPS link starting with "https://".

Create environment:

```bash
$ conda create -n ood-robustness python=3.11
$ conda env update -n ood-robustness --file environment.yaml
```

### pre-commit usage explicit (for contributors only)

```bash
$ pre-commit run -a
```

### pre-commit usage during git commit

Once after clone the replository run:

```bash
$ pre-commit install
```

Use [ConventionalCommits](https://www.conventionalcommits.org) syntax for commit messages.

```bash
$ git commit -am "feat: this is a message"
```

Install as package (optional):

```bash
$ python -m pip install .
```

Adding the project directory to the PYTHONPATH works as well.

## Usage

<a name="usage"></a>
### Prepare data
For preparing all dataset and model weight, we prepare a download script.
Although default values should work out of the box, data paths can be adapted, by either using comandline arguments or setting the corresponding environment variable.
We encourage users to create a `.env` file from the provided [.env.example](./.env.example) file, to configure environment variables.
```bash
python prepare_data.py
```

### Run experiments
Next step is to run the experiments using:
```bash
python run_experiment.py
```
This runs the experiments that were demonstrated in the manuscript.
If you are not using default dataset and model paths, make sure that the environment variable are set properly.
The script stores the results of the experiments in the [output](./output) directory.

### Visualize results
In order to visualize the results, we provide a Jupyter notebook [vis_results.ipynb](./notebooks/vis_results.ipynb).
The notebook load the results as stored by the [run_exeriment.py](./run_exeriment.py) scriped and generates interpretable plots for them.

## Contact

[Christian Huber](mailto:christian.huber@silicon-austria.com)

## License

See [MIT license](./LICENSE)

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) project template.
