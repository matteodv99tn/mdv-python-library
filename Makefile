doc: clean
	PYTHONPATH=. sphinx-apidoc -o docs/_autodoc mdv
	PYTHONPATH=. sphinx-autogen docs/index.rst
	PYTHONPATH=. sphinx-build docs html

clean:
	@echo "Removing files"
	@rm html/ docs/generated docs/_autodoc -r 2>>/dev/null || true
