#!/bin/bash

# Global variables
export model_path="/home/gridsan/yakiyama/AF_inv_covariance/model_checkpoints/train_presoftmax_diff_attn_04102025_fullft_no_adversarial/model_step_18000.pt"
export contact_model_path="/home/gridsan/yakiyama/AF_inv_covariance/model_checkpoints/train_contact_presoftmax_diff_attn_04142025/model_final.pt"
export dms_input="/home/gridsan/yakiyama/AF_inv_covariance/misc_data/proteingym/DMS_ProteinGym_substitutions/"
export dms_mapping="/home/gridsan/yakiyama/ProteinGym/reference_files/DMS_substitutions.csv"
export mutation_col="mutant"
export dms_output="results/"
export offset_idx=1
export msa_path="/home/gridsan/yakiyama/AF_inv_covariance/misc_data/proteingym/DMS_msa_files/"
export msa_weights_folder="msa_weights"
export path_to_hhfilter="/home/gridsan/yakiyama/other_tools/hhfilter/bin/hhfilter"
export seeds="0"
export sampling_strategy="sequence-reweighting"
export nseq=400
export scoring_strategy="masked-marginals"
export scoring_window="optimal"
export num_cpus=48

# Function to process a single DMS index
for dms_idx in {175..175}; do
    python compute_fitness.py \
        --model_path $model_path \
        --contact_model_path $contact_model_path \
        --dms-input $dms_input \
        --dms-index $dms_idx \
        --dms_mapping $dms_mapping \
        --mutation-col $mutation_col \
        --dms-output $dms_output \
        --offset-idx $offset_idx \
        --scoring-strategy $scoring_strategy \
        --msa-path $msa_path \
        --msa-sampling-strategy $sampling_strategy \
        --msa-samples $nseq \
        --msa-weights-folder $msa_weights_folder \
        --scoring-window $scoring_window \
        --overwrite-prior-scores \
        --seeds $seeds \
        --num-cpus $num_cpus
done
