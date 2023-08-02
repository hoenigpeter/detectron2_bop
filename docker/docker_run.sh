#!/bin/bash

# prepare /datasets, /pretrained_models and /output folders as explained in the main README.md
docker run \
--gpus all \
-it \
--shm-size=8gb --env="DISPLAY" \
--volume="/home/peter/detectron2_bop:/detectron2_bop" \
--name=detectron2_bopv0 detectron2_bop