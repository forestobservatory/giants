.PHONY: docs

test:
	pytest --cov=giants --no-cov-on-fail --cov-report=term-missing:skip-covered --cov-report xml:coverage.xml

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
