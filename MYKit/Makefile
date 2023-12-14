envname=274A_final
RUNWITHCONDA=conda run -n $(envname)


environment: environment.yml
	if conda info --envs | grep -q $(envname); then echo "$(envname) already exists"; else conda env create --file environment.yml; fi

test: environment
	$(RUNWITHCONDA) pytest -v --color=yes


lint: 
	$(RUNWITHCONDA) black  .
	$(RUNWITHCONDA) flake8 .
