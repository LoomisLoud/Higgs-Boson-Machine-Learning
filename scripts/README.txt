===========================================================================

This is our implementation of the Higg's Boson search (PCML, EPFL), the aim is to find Higg's
Boson, or at least to get the closest possible to it. This repository
contains all the code needed to reproduce our best score, but also to
conduct much more experiments, like running any other regression method
than the one we used ourselves.

If you did not yet read our report, do not forget that this code must be
run on a 16GB RAM *nix machine at least. We explain the tradeoff in the
report extensively.

You will find in the build_polynomial file all of the possible polynomials
you can use to run the project, in implementations, our machine learning
methods, in standardization the way we standardized our matrices, and
finally in run.py the bulk of the project, the actual process to obtain
the same predictions we got in the leaderboard.


In order to run this project, you need to:

  1. Put in the same folder as the run.py file, the train and test datasets
  renamed to: "train.csv" and "test.csv"

  2. Run the following command: "python run.py"
  (Assuming you are running python3 as default, otherwise, run
  "python3.5 run.py" instead)

  3. The output predictions will be in the "predictions.csv" file.

===========================================================================
