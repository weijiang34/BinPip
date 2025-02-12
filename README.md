# Binning Pipeline User Guide

This document provides a comprehensive guide on how to use the Binning Pipeline, a command-line software designed for various bioinformatics tasks including grouping, mapping, binning, and more. The pipeline is built to handle large datasets and provide detailed reports on the progress and outcomes of each step. The prototype of this pipeline was first introduced in: https://mrcbioinfo.github.io/mima-pipeline/, and was later refined and automated using Python scripts, as demonstrated here.

## Table of Contents

- [Binning Pipeline User Guide](#binning-pipeline-user-guide)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Modules](#modules)
    - [Grouping](#grouping)
    - [Mapping](#mapping)
    - [Binning](#binning)
    - [CheckM](#checkm)
    - [Check](#check)
    - [Filter](#filter)
    - [Classify](#classify)
    - [Quantify](#quantify)
  - [Examples](#examples)
  - [Troubleshooting](#troubleshooting)
  - [Contact](#contact)

## Overview

The Binning Pipeline is a versatile tool that streamlines the process of handling biological data. It offers multiple modules to perform specific tasks, allowing users to conduct binning analysis following the oder of modules. 

## Installation

- **Before installing dependencies, please ensure that the environment names match exactly those mentioned in the document.**

1. **Download pipeline script:**  
  To use the Binning Pipeline, ensure you have Python installed on your system. Then, clone this repository to your local system:
    ```
    git clone https://github.com/weijiang34/BinPip.git
    ```
2. **Install dependencies:**  
  The pipeline requires a series of dependent tools/softwares/scripts. Please follow the instructions below to install these dependencies.  
  - Miniconda/Conda  
    Please refer to the miniconda installation instructions: https://docs.anaconda.com/miniconda/miniconda-install/
  - MicroMamba/Mamba  
    Please refer to the micromamba installation instructions: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
  - SemiBin2
    ```
    conda create -y -n bar python=3.10
    conda activate bar
    conda install -c conda-forge -c bioconda semibin bowtie2 samtools
    ```
    #(try to use ```conda config --set channel_priority``` false if your environment can't be solved)  
    #(try to use ```mamba install -c conda-forge``` ... if everything else fails )
  - CheckM2  
    https://github.com/chklovski/CheckM2
    ```
    conda create -n checkm2 -c bioconda -c conda-forge checkm2
    ```
  - dRep  
    https://github.com/MrOlm/drep
    ```
    mamba create -n drep dRep
    ```
  - GTDB-TK  
    https://ecogenomics.github.io/GTDBTk/installing/bioconda.html
    ```
    conda create -n gtdbtk-2.1.1 -c conda-forge -c bioconda gtdbtk
    ```
  - MetaWrap  
    https://github.com/bxlab/metaWRAP
    ```
    git clone https://github.com/bxlab/metaWRAP.git
    ```
  - Salmon and python2.7   
    https://salmon.readthedocs.io/en/latest/index.html
    ```
    conda create -n salmon salmon python=2.7 numpy
    ```
3. **Configure environment variables**  
  Modify the following part according to your paths.


## Modules

### Grouping

This module groups samples by age and creates necessary directories.

**Usage:**

```bash
python /path/to/binning_pipeline.py -p /path/to/project grouping -m manifest.csv -a age_gender.csv -n 4
```

- `-p`: Project directory. (Default: ./)
- `-m`: Manifest file (CSV) with columns: fileHeader, fq1, fq2, fa.
- `-a`: Age and gender information file (CSV) with columns: sample_id, age, sexbirth.
- `-n`: Number of groups to split the data into (default: 4).

### Mapping

This module handles the concatenation of sequences, indexing, and mapping reads to contigs.

**Usage:**

```bash
python /path/to/binning_pipeline.py -p /path/to/project mapping
```

### Binning

This module generates sequence files, performs self-training, and binning.

**Usage:**

```bash
python /path/to/binning_pipeline.py -p /path/to/project binning
```

### CheckM

This module provides quality reports for each sample.

**Usage:**

```bash
python /path/to/binning_pipeline.py -p /path/to/project checkm
```

### Check

This module checks the completeness of each step in the pipeline.

**Usage:**

```bash
python /path/to/binning_pipeline.py -p /path/to/project check -nc 0.3 -sa 0.98
```

- `-nc`: Nucleotide coverage for dereplication (default: 0.3).
- `-sa`: Average nucleotide identity for dereplication (default: 0.98).

### Filter

This module selects genomes and performs dereplication.

**Usage:**

```bash
python /path/to/binning_pipeline.py -p /path/to/project filter -nc 0.3 -sa 0.98
```

### Classify

This module classifies MAGs using GTDB-TK.

**Usage:**

```bash
python /path/to/binning_pipeline.py -p /path/to/project classify -nc 0.3 -sa 0.98
```

### Quantify

This module quantifies MAGs using Salmon.

**Usage:**

```bash
python /path/to/binning_pipeline.py -p /path/to/project quant -nc 0.3 -sa 0.98
```

## Examples

Here are some example commands to get you started:

```bash
# Grouping samples
python /path/to/binning_pipeline.py -p /path/to/project grouping -m manifest.csv -a age_gender.csv -n 4

# Mapping
python /path/to/binning_pipeline.py -p /path/to/project mapping

# Binning
python /path/to/binning_pipeline.py -p /path/to/project binning

# CheckM
python /path/to/binning_pipeline.py -p /path/to/project checkm

# Check completeness
python /path/to/binning_pipeline.py -p /path/to/project check -nc 0.3 -sa 0.98

# Filter
python /path/to/binning_pipeline.py -p /path/to/project filter -nc 0.3 -sa 0.98

# Classify
python /path/to/binning_pipeline.py -p /path/to/project classify -nc 0.3 -sa 0.98

# Quantify
python /path/to/binning_pipeline.py -p /path/to/project quant -nc 0.3 -sa 0.98
```
Every time you generated some job files, please **double check and manually submit** them. 

## Troubleshooting

If you encounter any issues, ensure that all dependencies are installed and that your input files are formatted correctly. For further assistance, please refer to the contact information below.

## Contact

For any questions or support, please contact:

- Email: [wjiang34-c@my.cityu.edu.hk]
