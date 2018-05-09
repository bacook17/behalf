default: gpu

gpu: behalf_gpu

cpu: behalf_cpu

cy:
	python setup.py build_ext --inplace clean

behalf_only: cy
	@echo "---------------------------------------"
	@echo "installing behalf only, NO dependencies"
	pip install . --user --upgrade --no-deps
	@echo "successfully completed installing behalf with NO dependencies"
	@echo "---------------------------------------"

behalf_cpu: cy
	@echo "---------------------------------------"
	@echo "installing behalf with CPU support only"
	pip install . --user --upgrade
	@echo "successfully completed installing behalf with CPU support only"
	@echo "---------------------------------------"

behalf_gpu: cy behalf_cpu
	@echo "---------------------------------------"
	@echo "attempting to install GPU support for behalf"
	@pip install -q .[GPU] --user --upgrade || (echo "GPU support not available. Confirm NVIDIA GPU is available, the NVIDIA driver and CUDA are installed."; exit 1)
	@echo "successfully completed installing behalf with GPU support"
	@echo "---------------------------------------"

manual: cy
	python setup.py install clean
