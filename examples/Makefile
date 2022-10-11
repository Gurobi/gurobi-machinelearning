run_examples_%:
	cd $*; echo "TimeLimit 20" > gurobi.env; jupyter nbconvert --to python *.ipynb; ipython --matplotlib pdf *.py

all: run_examples_adversarial run_examples_avocado run_examples_Functions_approx run_examples_Janos

clean:
	rm -f */gurobi.env
	rm -f */*.py

.PHONY: clean all
