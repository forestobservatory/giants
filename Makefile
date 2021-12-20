NAME=giants
CONDA=conda run --name ${NAME}

.DEFAULT: help
help:
		@echo "--- [ $(NAME) developer tools ] ---"
	  @echo ""
	  @echo "make init - initialize conda dev environment"
	  @echo "make test - run package tests"

init:
		conda env list | grep -q ${NAME} || conda create --name=${NAME} python=3.7 -y
	  ${CONDA} pip install -e .
	  ${CONDA} pip install -r requirements-dev.txt
	  ${CONDA} conda install pre-commit -c conda-forge && ${CONDA} pre-commit install

test:
		pytest --cov=giants --no-cov-on-fail --cov-report=term-missing:skip-covered --cov-report xml:coverage.xml

# deprecate everything below once CI is properly setup
coverage:
		./codecov

docs:
		mkdocs gh-deploy

package:
		python3 setup.py sdist bdist_wheel
		twine upload dist/*

all:
		test
		coverage
		docs
		package
