
DOCS_SOURCES=../doc_source/source

run_examples_%:
	cd $*; echo "TimeLimit 20" > gurobi.env; jupyter nbconvert --to python *.ipynb; ipython --matplotlib pdf *.py

run_quickstart:
	cd $(DOCS_SOURCES); jupyter nbconvert --to python *.ipynb; ipython --matplotlib pdf *.py

run_doc_example:
	cd $(DOCS_SOURCES)/examples; jupyter nbconvert --to python *.ipynb; ipython --matplotlib pdf *.py

all: run_examples_adversarial run_examples_avocado run_examples_Functions_approx run_examples_Janos \
	   run_quickstart run_doc_example_adversarial run_doc_example

clean:
	rm -f */gurobi.env
	rm -f */*.py

.PHONY: clean all
