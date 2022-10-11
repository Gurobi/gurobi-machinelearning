run_examples_%:
	cd $*; echo "TimeLimit 20" > gurobi.env; jupyter nbconvert --to python *.ipynb; ipython --matplotlib pdf *.py

all:
	make run_examples_adversarial
	make run_examples_avocado
	make run_examples_Functions_approx
	make run_examples_Janos
