MSc project: Interpretable belief dynamics in neural representations — preprocessing, decoding, saliency, and dynamical model discovery.

### Setup:
make sure conda-lock and mamba is installed in base
```powershell
conda install -y -c conda-forge mamba
```
create mamba env
```powershell
mamba env create --override-channels -f environment.yml
```
activate mamba env
```powershell
mamba activate my_msc_proj
```
install pytorch
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
install project to fix src default imports
```powershell
pip install -e .
```

clone the IBL_MtM
```powershell
git clone https://github.com/colehurwitz/IBL_MtM_model.git
```
install IBL_MtM without its environment
```powershell
pip install --no-deps -e .\IBL_MtM_model\src
```



### Usage:
create local mlflow server for online model training tracking
```powershell
mlflow server --port 5000 --backend-store-uri ./mlruns
```
