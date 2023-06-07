.PHONY: help build clean

help:
	@echo "Available targets:"
	@echo "  clean-notebooks  : Clean up notebooks and their output (only affects the `notebooks/` folder)"

clean-notebooks:
	@echo "Cleaning up notebooks and their output..."

	rm -rf ./notebooks/**/**/.ipynb_checkpoints

	find ./notebooks -name "*.ipynb" -type f -print0 | xargs -0 -I {} jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {}
