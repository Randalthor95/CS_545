# CS_545
Repo for coursework for CSU CS 545 https://www.cs.colostate.edu/~anderson/cs545/doku.php?id=schedule

CS 545: Machine Learning

D:
cd D:\College\CS_545\Assignments
conda activate ml_assignments
jupyter lab

D:
cd D:\College\CS_545\Lessons
conda activate ml
jupyter lab

D:
cd D:\College\CS_545\Final_Project
conda activate fp
jupyter lab

cd C:\Users\saras\Documents\Tony\CS_545\Assignments
conda activate ml_assignments
jupyter lab

cd C:\Users\saras\Documents\Tony\CS_545\Lessons
conda activate ml
jupyter lab

conda env update --file environment.yml --prune

conda env create -f environment.yml

python -m ipykernel install --user --name=ml_assignments



https://www.cs.colostate.edu/machinestats/

https://www.cs.colostate.edu/~info/machines

ssh acf003@bentley.cs.colostate.edu

export PATH="/usr/local/anaconda3/latest/bin:$PATH"
export PYTHONPATH="/usr/local/anaconda3/latest/lib/python3.8/site-packages:$PYTHONPATH"

cd CS_545
conda activate ml

jupyter-notebook --no-browser --port=8080 > cmd_output.txt

ssh -N -L 8080:localhost:8080 acf003@bentley.cs.colostate.edu


echo "${PATH//:/$'\n'}"

pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/cuda111/jaxlib-0.1.67+cuda111-cp38-none-manylinux2010_x86_64.whl