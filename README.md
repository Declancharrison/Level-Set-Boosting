# What is it?

lsboost is a regression boosting algorithm for multicalibration defined in ([Globus-Harris et al. 2023](https://arxiv.org/pdf/2301.13767.pdf)). Multicalibration is a realtively new notion of fair machine learning designed to ensure that identified subgroups of the population in your data do not receive predictions which are far away from their conditional label mean. 

# Download

This package is not yet available for installation via pip and thus must be downloaed from this repository. You can easily download the repository by clicking on the green code button and following the GitHub instructions listed. The following command can be run in the terminal for example:
~~~bash
git clone https://github.com/Declancharrison/Level-Set-Boosting.git
~~~

# Usage

A notebook titled **LSBoost_notebook.ipynb** has been provided to give an example for using lsboost on census data from the [Folktables](https://github.com/socialfoundations/folktables) package. Further descriptions of hyperparameters and their uses can be found in the init for the class *LSBoostingRegressor* in *LSBoost.py*.

# Citing lsboost

If you use lsboost, please cite the paper it originates from:


 <!-- @inproceedings{globusharris2023multicalibration,
    title = {Multicalibration as Boosting for Regression},
    author = {Ira Globus-Harris and Declan Harrison and Michael Kearns and Aaron Roth and Jessica Sorrell},
    booktitle = {Proceedings of the 40th International Conference on Machine Learning},
    pages = {1939--1948},
    year = {2023},
    editor = {Jennifer Dy and Andreas Krause},
    volume = {80},
    series = {Proceedings of Machine Learning Research},
    address = {Stockholmsmässan, Stockholm Sweden},
    publisher = {PMLR}
  } -->
@misc{globusharris2023multicalibration,
      title={Multicalibration as Boosting for Regression}, 
      author={Ira Globus-Harris and Declan Harrison and Michael Kearns and Aaron Roth and Jessica Sorrell},
      year={2023},
      eprint={2301.13767},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}