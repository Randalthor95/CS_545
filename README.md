# CS_545
Repo for coursework for CSU CS 545 https://www.cs.colostate.edu/~anderson/cs545/doku.php?id=schedule

CS 545: Machine Learning

D:
cd D:\College\CS_545\Lessons
conda activate ml
jupyter lab

D:
cd D:\College\CS_545\Assignments
conda activate ml_assignments
jupyter lab


conda env update --file environment.yml --prune

conda env create -f environment.yml

python -m ipykernel install --user --name=ml_assignments
