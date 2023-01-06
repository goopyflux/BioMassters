###############################################################################
# Makefile for building Docker images and running Docker containers 
# with pyenv, pdm, PyData Stack, JupyterLab + more
# See documentation in Dockerfile for more information.
#
# Maintainer: Goopy Flux <goopy.flux@gmail.com>
###############################################################################

# Note: Update the image tag if the Dockerfile changes
# Local Image
image_name = gradient-fastai-aws
image_tag = dev-local
local_image = ${image_name}:${image_tag}

# Remote Repository on Docker Hub
docker_hub_repo = goopyflux
remote_image = ${docker_hub_repo}/${image_name}:${image_tag}

python_version = 3.9.13
dockerfile = Dockerfile

# #############
# make commands
# #############

# Host volume to mount
host_volume ?= ${PWD}
container_name = biomasstery

# Note: delete the --rm option, if you wish to persist the container upon exit.
# Ex. may be to call `docker commit` to save the container as a new image.
## Run the JupyterLab Docker container. Use host_volume to specify local folder.
## (Ex. make docker-run host_volume=/home/user/work)
docker-run:
	docker run -it --init -p 8888:8888 -v "${host_volume}:/notebooks" --name ${container_name} ${local_image}

## Start an existing container
docker-start:
	docker container start ${container_name}

## Open a bash shell on a running container
docker-exec:
	docker exec -it ${container_name} /bin/bash

## Setup Paperspace Gradient Notebook
setup-gradient:
	pip install --upgrade s3fs rasterio && \
	pip install --upgrade transformers accelerate nvidia-ml-py3 && \
	pip install -e .

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

