# Make all targets .PHONY
.PHONY: $(shell sed -n -e '/^$$/ { n ; /^[^ .\#][^ ]*:/ { s/:.*$$// ; p ; } ; }' $(MAKEFILE_LIST))

SHELL = /usr/bin/env bash

NOTEBOOK_FILE=/src/encoder/open_clip_encoder.ipynb
NOTEBOOK_TEMPLATE=./src/config
KAGGLE_URL=https://www.kaggle.com/code

export 


create-and-run-kaggle-notebook:
	@echo "⏳ Creating Kaggle notebook..."
	kaggle kernels push -p $(NOTEBOOK_TEMPLATE)
	@if [ "$(OS)" = "Windows_NT" ]; then \
		start $(KAGGLE_URL); \
	else \
		xdg-open $(KAGGLE_URL) || open $(KAGGLE_URL); \
	fi

data-versioning-first: 
	rm -rf .dvc/
	poetry run python -m src.data.task_version_data
	
data-versioning: 
	@read -p "Do you run for the first time: y or n? " choice; \
	case $$choice in \
		y|Y) make data-versioning-first;; \
		n|N) poetry run python -m src.data.task_version_data;; \
		*) echo "Invalid choice. Please enter 'y' or 'n'.";; \
	esac

test:
	@echo "Choose a test function to run:"
	@echo "1. test_text_search"
	@echo "2. test_ocr_search"
	@echo "3. test_image_search"
	@read -p 'Enter the function number (1-3): ' func; \
	case $$func in \
		1) make run-test FUNCTION="test_text_search" ;; \
		2) make run-test FUNCTION="test_ocr_search" ;; \
		3) make run-test FUNCTION="test_image_search" ;; \
		*) echo "Invalid choice. Please select a valid function."; exit 1;; \
	esac

run-test:
	@echo "Running $$FUNCTION test..."
	pytest test_function.py::$$FUNCTION
	
app-local:
	poetry run uvicorn app.application:app --host 0.0.0.0 --port 8000

pipeline:
	poetry run python all.py

menu: 
	@echo "Choose an action:"
	@echo "1. Data versioning"
	@echo "2. Test Function"
	@echo "3. Run app local (FastAPI)"
	@read -p "Enter a number (1-5): " choice; \
	case $$choice in \
		1) make data-versioning ;; \
		2) make test ;; \
		3) make app-local ;; \
	esac