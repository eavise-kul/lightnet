SHELL := /bin/bash
DEBUG := 
all: test docs
.PHONY: docker docs doctest unittest lint test build
.SILENT: docker docs doctest test build
.NOTPARALLEL: docker docs doctest unittest lint test build

docker: name := TOP_lightnet
docker: image := top/pytorch
docker:
	${DEBUG} NV_GPU=1 nvidia-docker run -it --rm --name ${name} -h ${name} -e TERM \
		-e DISPLAY --net=host --ipc=host \
    -v ${PWD}:/developer/project \
    ${image} /bin/bash -c \
    "cd project; \
		pip install -r develop.txt; \
    clear; \
    /bin/bash"

docs: notebook:= 0
docs:
	${DEBUG} cd ./docs && ${DEBUG} JNB=${notebook} make clean html

doctest:
	${DEBUG} cd docs; ${DEBUG} make doctest

unittest: file := ./test/
unittest: flag := 
unittest:
	BB_LOGLVL=warning python -m pytest ${flag} ${file}

lint:
	${DEBUG} pycodestyle lightnet/
	${DEBUG} pycodestyle test/

test: unittest doctest lint

build: test
	rm -rf dist/*
	python setup.py sdist bdist_wheel
	for w in dist/*.whl; do auditwheel repair $$w; done
	echo -e '\n \033[1mUpload your package to (test) PyPi\e[0m'
