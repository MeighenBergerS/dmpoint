# dmpoint

Authors:

1. Stephan Meighen-Berger

## Table of contents

1. [Introduction](#Introduction)
2. [Data](#Data)
3. [Guide](#Guide)
4. [Citation](#Citation)

## Introduction <a name="Introduction"></a>

Project to check if we can constrain dm using point source approaches

## Data <a name="Data"></a>

This code was designed to use data from the public IceCube 10 year point source
data release:

IceCube Collaboration (2021): All-sky point-source IceCube data: years 2008-2018. Dataset.
DOI: <http://doi.org/DOI:10.21234/sxvs-mt83>

## Guide <a name="Guide"></a>

The repository includes all scripts required to generate the plots from the
publication, except the atmospheric shower one. For that please use
[MCEq](https://github.com/afedynitch/MCEq).

The jupyter notebooks should guide through the required steps to analyze the data.
(As of yet not designed to be user-friendly. Given interest, this will be changed)

1. sky_hotspots_density.ipynb: generates the density maps given the IceCube data events
2. sky_hotspots_density_signal.ipynb: generates the density maps for a given signal
3. sky_hotspots_kmean.ipynb: performs a kmean test on the data set
4. cluster_reader.ipynb: Reads cluster generated data from the codes in the cluster folder

The cluster folder contains scripts to run batches on a cluster using slurm.

## Citation <a name="Citation"></a>

Please cite [arXiv:2109.07885](https://arxiv.org/abs/2109.07885) when using this code.
