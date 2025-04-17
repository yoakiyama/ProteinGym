#!/bin/bash

# for dms_idx in {184..185}; do
for dms_idx in {140..183}; do
    dms_idx=$dms_idx LLsub compute_fitness_llsub.sh -g volta:1
done