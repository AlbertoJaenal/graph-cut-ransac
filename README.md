<div align="center">
<h1>GC-RANSAC implementation of Gravity-aware partially calibrated absolute <br> pose estimation from affine- or <br> rotation-covariant features</h1>
<a href="_blank"><img src="https://img.shields.io/badge/Paper-blue" alt="Paper"></a>
<a href="https://marcusvaltonen.github.io/eccv2026/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<br>
<br>
<strong>
<a href="https://scholar.google.com/citations?user=U_U-GmcAAAAJ">Marcus Valtonen Örnhag<sup>1</sup></a>
&nbsp;&nbsp;
<a href="https://scholar.google.com/citations?user=UYL0UY0AAAAJ">Alberto Jaenal<sup>2</sup></a>
&nbsp;&nbsp;
<a href="https://scholar.google.com/citations?user=jLJ6ZaMAAAAJ">Stefan Adalbjörnsson<sup>1</sup></a>
<br>
<br>
<sup>1</sup> Ericsson Research &nbsp;&nbsp;
<sup>2</sup> University of Zaragoza
</strong>
<br>
<br>
<br>
</div>

GCRANSAC implementation for the ECCV 2026 spotlight paper.

## Overview

This repo implements the experiments on real data presented in the paper. For that, we implement our [solvers](https://github.com/marcusvaltonen/eccv2026/tree/main) undistorted camera absolute regression problem is *in GCRANSAC** (Barath and Matas, 2018, [original repo](https://github.com/danini/graph-cut-ransac)), so that it can be easily integrated in a pipeline with *HLoc*. The GCRANSAC is compiled through C++ and binded to the **pygcransac** library, which is used in the tests.

Implemented solvers (see [here](testing_absolute/utils/absolute_defs.py), and see paper for corresponding article):
* Calibrated: P3P, UP2P_ERI, P1P_AC, UP1P_SIFT
* Uncalibrated: P4PF, P35PF, UP3PF, UP1PF_AC (ours), UP2PFORI (ours)


## Requirements

- Python 3
- CMake 2.8.12 or higher
- OpenCV 3.4
- A modern compiler with C++11 support
- HLoc and Open3d (see requirements.txt)


## Installation C++

```shell
git clone https://github.com/AlbertoJaenal/graph-cut-ransac
git checkout eccv26
git submodule update --init --recursive

sudo apt install libopencv-dev libeigen3-dev
pip install -r requirements.txt

bash build_deps.sh
cd build
cmake .. -DCMAKE_CXX_FLAGS=-w ..
make
```

# Run experimente

A [simple example](testing_absolute/example_test_absolute.py) is set for sandbox testing, based on Reichstag from Phototourism. The folder is downloaded in the datasets folder. It reconstructs the scene from a train split, extracts the affine features and finally tests with all the solvers.
```
cd testing_absolute/
python example_test_absolute.py 
```

For the papers tested in Cambridge and Aachen datasets. Once installed, for the real experiments, see [testing_absolute](testing_absolute/) folder. Download the datasets wherever you want, change the data path [here](testing_absolute/utils/__init__.py). Example using two different feature configurations (SIFT or SuperPoint+LightGlue):
```
cd testing_absolute/
# For Cambridge
python eccv26_pipeline_cambridge.py --feature=(sift)|(sp+lg)
# For Aachen
python eccv26_pipeline_aachen.py --feature=(sift)|(sp+lg)

# For more options
python eccv26_pipeline_aachen.py --help
```

The absolute pose estimation process is parellelized, so that it can run faster. Just change the `parallel=-1`to run it in one process (note, Aachen may take long time in this case).

# Acknowledgemetns

We based this code on the GC-RANSAC, PoseLib and the HLoc libraries.

# Citing

When using this work, please cite
```
@InProceedings{valtonen-ornhag-jaenal-2026-eccv,
    author    = {Valtonen~{\"O}rnhag, Marcus and Jaenal, Alberto and Adalbj{\"o}rnsson, Stefan},
    title     = {Gravity-aware partially calibrated absolute pose estimation
from affine- or rotation-covariant features},
    booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
    year      = {2026},
}

```

When using the GC-RANSAC algorithm, please cite

```
@inproceedings{GCRansac2018,
	author = {Barath, Daniel and Matas, Jiri},
	title = {Graph-cut {RANSAC}},
	booktitle = {Conference on Computer Vision and Pattern Recognition},
	year = {2018},
}

```
