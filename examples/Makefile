all: run_examples_adversarial run_examples_avocado run_examples_Functions_approx run_examples_Janos \
	 docs_examples

run_examples_%:
	cd $*; echo "TimeLimit 20" > gurobi.env; jupyter nbconvert --to python *.ipynb; ipython --matplotlib pdf *.py

docs_examples:
	cd ../doc_source/notebooks/ipynb; jupyter nbconvert --to python *.ipynb; ipython --matplotlib pdf *.py

clean:
	rm -f */gurobi.env
	rm -f */*.py

.PHONY: clean all
