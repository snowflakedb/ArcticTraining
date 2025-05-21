# usage: make help

.PHONY: help test test-cpu test-gpu format
.DEFAULT_GOAL := help

help: ## this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[0-9a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	echo $(MAKEFILE_LIST)

test: ## run cpu and gpu tests
	pytest --disable-warnings --instafail ./tests/

test-cpu: ## run cpu-only tests
	pytest --disable-warnings --instafail -m "not gpu" ./tests/

test-gpu: ## run gpu-only tests
	pytest --disable-warnings --instafail -m gpu ./tests/

format: ## fix formatting
	@if [ ! -d "venv" ]; then \
		sudo apt update; \
		sudo apt-get install -y python3-venv; \
		python -m venv venv; \
		. venv/bin/activate; \
		pip install pre-commit -U; \
		pre-commit clean; \
		pre-commit uninstall; \
		pre-commit install; \
		deactivate; \
	fi
	. venv/bin/activate && pre-commit run --all-files && deactivate

# this tool is optional not to be run automatically as it could have unexpected side-effects, but is useful when
# needing to remove a bulk of unused imports
autoflake: ## autoremove unused imports (careful!)
	@read -p "Running autoflake will remove unused imports and modify files in place. This could have unexpected side-effects. Do you want to continue? [y/n] " ans; \
	if [ "$$ans" != "y" ]; then \
		echo "Aborted."; \
		exit 1; \
	fi; \
	autoflake --verbose --in-place --remove-all-unused-imports --ignore-init-module-imports --ignore-pass-after-docstring -r arctic_training
