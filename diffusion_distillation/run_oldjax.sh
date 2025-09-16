#~/bin/bash

# # Download the diffusion_distillation repository
git clone --filter=blob:none -b master --no-checkout https://github.com/xvdp/google-research.git
cd google-research
git sparse-checkout set diffusion_distillation
git checkout
# pip install -r diffusion_distillation/diffusion_distillation/requirements.txt --quiet
# this is installed already in Dockerfile.oldjax

