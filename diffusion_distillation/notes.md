

# Diffusion Distillation
Suffers from deprecations between google colab ( which runs on latest numpy jax flax jaxlib optax ) and methods developed in this paper  using older versions **early 2022** of the those. At most
```
python 3.10 
flax 0.4.1
jax 0.4.1
jaxlib 0.4.1 # was yanked from
pip install jaxlib==0.4.1 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
## Issues
* https://github.com/google/flax/issues/4944
* https://github.com/google-research/google-research/issues/2976

Fixed colab and model.py to use latest optax / flax / jax / 2025, but something is incorrect in loop.
* Distillation seems to work 
* **Training from scratch does not work:** 
https://github.com/xvdp/google-research/blob/fixflaxversion/diffusion_distillation/diffusion_distillation.ipynb

##
Built 2 dockerimages, 
1. `google_research/xvdp/diffusion_distillation:jax0.4.1`  from Dockerfile.oldjax. Could be used to debug `flax.optim` in flax==0.4.1 vs optax.` in latests code and ajust it. requires model from   https://github.com/xvdp/google-research/blob/master/diffusion_distillation/diffusion_distillation/model.py

Run without passing workdir or sharing diffusion distillation
```bash
docker run --gpus device=1 --cpuset-cpus=0-20 --network=host -it --rm --shm-size 20g diffusion_distillation:jax0.4.1

bash run_oldjax.sh # clones into master, 
cd google-research &&  python3
```
**==> uselss pursuit jax==0.4.1 no longer can run all the functions in the project. so stale dockerfile here. checked in just so i dont throw it away, but its useless** 

2. `google_research/xvdp/diffusion_distillation:latest`  from Docekrfile.
Should work on local gpus.