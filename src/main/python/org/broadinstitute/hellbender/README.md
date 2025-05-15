Python packages to be used in the GATK should be placed in this directory.  
Each package should be contained in its own subdirectory.  If the package 
can be installed as a standalone package, a corresponding `setup_<PACKAGE_NAME>.py` 
file may be placed in this directory.  However, during creation of the common 
GATK conda environment, all packages will be combined and pip-installed as a 
single package named ``gatkpythonpackages`` by `setup.py`.

However, note that it is easier to do development by installing live/editable versions of these packages
(i.e., running `pip install --editable .` in this directory), so that any code changes are immediately reflected in 
the underlying environment. To do this, 1) remove the pip install of the `gatkpythonpackages.zip` archive in the 
conda environment file, 2) create and activate the corresponding conda environment, then 
3) run the editable pip install.

Additional
pip install cyvcf2
pip install pytorch-lightning==2.4.0
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
