# Data prepration for CREDIT stage 1 projects

This repository contains scripts and nootbooks that produce 1 degree grid and CESM grid ERA5 data for the training of CREDIT models. This is part of the CREDIT stage one project at NSF NCAR.

# Overview

Stage one data is prepared under the following steps
* 0.25 degree ERA5 on pressure level and hybrid sigma-pressure level (model level) were downloaded from [NCAR/RDA storage](https://rda.ucar.edu/datasets/d633000/) , and [ARCO ERA5](https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5). This part is not included in this repo.
* 0.25 degree ERA5 datasets were interpolated to 1 degree grid and CESM grid using conservative mapping. This part is not included in this repo.
* The interpolated ERA5 is merged into yearly datasets with needs variables.
* Mean, std, and residual normalization coefficients were computed on the merged datasets.
* Data is accessble within NSF NCAR HPCs.

## Contact
Yingkai Sha <ksha@ucar.edu>
