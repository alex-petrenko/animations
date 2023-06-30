line_len = 120
line_len_arg = --line-length $(line_len)
code_folders = ./


# Format source code automatically
.PHONY: format

format:
	black $(line_len_arg) -t py38 $(code_folders)
	isort $(line_len_arg) --py 38 --profile black $(code_folders)


# Check that source code meets quality standards
.PHONY: check-codestyle

check-codestyle:
	black --check $(line_len_arg) -t py38 $(code_folders)
	isort --check-only $(line_len_arg) --py 38 --profile black $(code_folders)
# ignore some formatting issues already covered by black
	flake8 --max-line-length $(line_len) --ignore=E501,F401,E203,W503,E126,E722,F405,F403 $(code_folders)
