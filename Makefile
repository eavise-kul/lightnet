SHELL := /bin/bash
DEBUG := 
all: test docs
.PHONY: docs doctest unittest lint test build
.SILENT: docs doctest test build unittest
.NOTPARALLEL: docs doctest unittest lint test build

docs: notebook:= 0
docs:
	${DEBUG} cd ./docs && ${DEBUG} JNB=${notebook} make clean html

doctest:
	${DEBUG} cd docs; ${DEBUG} make doctest

unittest: file := ./test/
unittest: expr := 
unittest: marker :=
unittest: EXPR := $(if $(strip ${expr}), -k ${expr},)
unittest: MARKER := $(if $(strip ${marker}), -m ${marker},)
unittest:
	BB_LOGLVL=warning python -m pytest ${EXPR} ${MARKER} ${file}

lint:
	${DEBUG} pycodestyle lightnet/
	${DEBUG} pycodestyle test/

test: unittest doctest lint

build: test
	rm -rf dist/*
	python setup.py sdist bdist_wheel
	for w in dist/*.whl; do auditwheel repair $$w; done
	echo -e '\n \033[1mUpload your package to (test) PyPi\e[0m'
