import argparse
import pathlib
import os
import sys
import random
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
from Bio import SeqIO
from typing import List, Tuple
import torch
from torch.nn.functional import one_hot

from msa_model.modules import MSAContactModel
from msa_model.dataset import MSA, prepare_additional_molecule_feats, aa2tok_d, prepare_msa_masks
from compute_fitness_utils import *

def main():
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Load deep mutational scan
    mutant_col = args.mutation_col  # Default "mutant"
    # If index of DMS file is provided
    if args.dms_index is not None:
        mapping_protein_seq_DMS = pd.read_csv(args.dms_mapping)
        DMS_id = mapping_protein_seq_DMS["DMS_id"][args.dms_index]
        print("Compute scores for DMS: "+str(DMS_id))
        row = mapping_protein_seq_DMS[mapping_protein_seq_DMS["DMS_id"]==DMS_id]
        if len(row) == 0:
            raise ValueError("No mappings found for DMS: "+str(DMS_id))
        elif len(row) > 1:
            raise ValueError("Multiple mappings found for DMS: "+str(DMS_id))
        
        row = row.iloc[0]
        row = row.replace(np.nan, "")  # Makes it more manageable to use in strings

        args.sequence = row["target_seq"].upper()
        args.dms_input = str(args.dms_input)+os.sep+row["DMS_filename"]

        mutant_col = row["DMS_mutant_column"] if "DMS_mutant_column" in mapping_protein_seq_DMS.columns else mutant_col
        args.dms_output=str(args.dms_output)+os.sep+DMS_id+'.csv'
        
        target_seq_start_index = row["start_idx"] if "start_idx" in mapping_protein_seq_DMS.columns and row["start_idx"]!="" else 1
        target_seq_end_index = target_seq_start_index + len(args.sequence) 

        # Get MSA file paths and indices
        msa_filename = row["MSA_filename"]
        if msa_filename == "":
            raise ValueError("No MSA found for DMS: "+str(DMS_id))

        args.msa_path= str(args.msa_path)+os.sep+msa_filename # msa_path is expected to be the path to the directory where MSAs are located.
            
        msa_start_index = int(row["MSA_start"]) if "MSA_start" in mapping_protein_seq_DMS.columns else 1
        msa_end_index = int(row["MSA_end"]) if "MSA_end" in mapping_protein_seq_DMS.columns else len(args.sequence)
        
        MSA_weight_file_name = args.msa_weights_folder + os.sep + row["weight_file_name"] if ("weight_file_name" in mapping_protein_seq_DMS.columns and args.msa_weights_folder is not None) else None
        if ((target_seq_start_index!=msa_start_index) or (target_seq_end_index!=msa_end_index)):
            args.sequence = args.sequence[msa_start_index-1:msa_end_index]
            target_seq_start_index = msa_start_index
            target_seq_end_index = msa_end_index
        df = pd.read_csv(args.dms_input)
    # If no index of DMS file is provided
    else:
        DMS_id = str(args.dms_input).split(os.sep)[-1].split('.csv')[0]
        args.dms_output=str(args.dms_output)+os.sep+DMS_id+'.csv'
        target_seq_start_index = args.offset_idx
        args.sequence = args.target_seq.upper()
        if (args.MSA_start is None) or (args.MSA_end is None): 
            if args.msa_path: print("MSA start and end not provided -- Assuming the MSA is covering the full WT sequence")
            args.MSA_start = 1
            args.MSA_end = len(args.target_seq)
        msa_start_index = args.MSA_start
        msa_end_index = args.MSA_end
        MSA_weight_file_name = args.msa_weights_folder + os.sep + args.weight_file_name if args.msa_weights_folder is not None else None
        df = pd.read_csv(args.dms_input)
    # Check if the dataframe is empty
    if len(df) == 0:
        raise ValueError("No rows found in the dataframe")
    print(f"df shape: {df.shape}", flush=True)
    # Get all variant positions
    dms_positions = set(df[mutant_col].apply(lambda x: x.split("|")[1]))
    print(f"Number of DMS positions: {len(dms_positions)}", flush=True)

    # Load MSA model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_kwargs = dict(
        dim_msa_input = 28,
        dim_pairwise = 256,
        dim_msa = 464,
        dim_logits = 26,
        msa_module_kwargs = dict(
            depth = 22,
            opm_kwargs = dict(
                dim_opm_hidden = 16,
                outer_product_flavor = "presoftmax_differential_attention",
                seq_attn = True,
                dim_qk = 128,
                chunk_size = None,
                return_seq_weights = True,
                return_attn_logits = False,
                lambda_init = None,
                eps = 1e-32,
            ),
            pwa_kwargs = dict(
                heads = 8,
                dim_head = 32,
                dropout = 0.1,
                dropout_type = "row",
            ),
            pairwise_block_kwargs = dict(
                tri_mult_dim_hidden = None,
                use_triangle_attn = False,
                use_triangle_updates = True
            )
        ),
        relative_position_encoding_kwargs = dict(
            r_max = 32,
            s_max = 2,
        ),
        return_msa_repr = False,
        return_pairwise_repr = True,
        query_only = True
    )
    checkpoint_path = args.model_path
    model = MSAContactModel(pretrained_weights_path=checkpoint_path, pretrained_is_final=False, msa_model_kwargs=model_kwargs, compiled_checkpoint=True, dim_contact_hidden=128)

    # Load contact model weights
    contact_checkpoint_path = args.contact_model_path
    contact_checkpoint = torch.load(contact_checkpoint_path, weights_only=True)
    contact_state_dict = {k.replace("_orig_mod.", ""): v for k, v in contact_checkpoint.items() if "contact" in k}
    _, _ = model.load_state_dict(contact_state_dict, strict=False)
    model = model.to(device)
    model = model.eval()

    # Compute fitness scores
    print("Computing model scores...")
    args.offset_idx = msa_start_index
    # Process MSA
    processed_msa = process_msa(filename=args.msa_path, weight_filename=MSA_weight_file_name, filter_msa=args.filter_msa, hhfilter_min_cov=args.hhfilter_min_cov, hhfilter_max_seq_id=args.hhfilter_max_seq_id, hhfilter_min_seq_id=args.hhfilter_min_seq_id, path_to_hhfilter=args.path_to_hhfilter, num_cpus=args.num_cpus)
    if os.path.exists(args.dms_output):
        prior_score_df = pd.read_csv(args.dms_output)
        if not args.overwrite_prior_scores:
            print("Prior scores found -- skipping model scoring")
            exit()
        else:
            print("Overwriting prior scores")
    else:
        prior_score_df = None
    seed = args.seeds[0]
    data = [sample_msa(sampling_strategy=args.msa_sampling_strategy, filename=args.msa_path, nseq=args.msa_samples, weight_filename=MSA_weight_file_name, processed_msa=processed_msa, random_seed=seed, num_cpus=args.num_cpus)]
    assert (args.scoring_strategy in ["masked-marginals","pseudo-ppl"]), "Zero-shot scoring strategy not supported with MSA Transformer"

    # Prepare data for MSA model
    nTokenTypes = len(np.unique(list(aa2tok_d.values())))
    msa_onehot_t, tokenized_msa_t, mask, msa_mask, full_mask, pairwise_mask, additional_feats_t = prepare_msa_inputs(data)
    mask, msa_mask, full_mask, pairwise_mask, additional_feats_t = mask.to(device), msa_mask.to(device), full_mask.to(device), pairwise_mask.to(device), additional_feats_t.to(device)

    # Compute masked-marginals scores
    if args.scoring_strategy == "masked-marginals":
        all_token_probs = []
        for i in tqdm(range(tokenized_msa_t.size(1)), desc="Scoring masked-marginals"):
            # Skip if position is not in DMS
            if i+1 not in dms_positions:
                zero_t = torch.zeros((1, 26))
                all_token_probs.append(zero_t)
                continue
            tokenized_msa_masked_t = tokenized_msa_t.clone()
            tokenized_msa_masked_t[0, i] = aa2tok_d['MASK']  # mask out first sequence
            if tokenized_msa_t.size(-1) > 1024:
                large_tokenized_msa_masked_t=tokenized_msa_masked_t.clone()
                start, end = get_optimal_window(mutation_position_relative=i, seq_len_wo_special=len(args.sequence)+2, model_window=1024)
                print("Start index {} - end index {}".format(start,end))
                tokenized_msa_masked_t = large_tokenized_msa_masked_t[:,:,start:end]
            else:
                start=0
            with torch.no_grad():
                msa_masked_onehot_t = one_hot(tokenized_msa_masked_t, num_classes=nTokenTypes).float().unsqueeze(0).to(device)
                token_probs = torch.log_softmax(
                    model(
                        additional_molecule_feats = additional_feats_t,
                        msa = msa_masked_onehot_t,
                        mask = mask,
                        msa_mask = msa_mask,
                        pairwise_mask = pairwise_mask,
                        full_mask = full_mask
                    )['logits'], dim=-1
                )
            all_token_probs.append(token_probs[:, 0, i-start].detach().cpu())  # vocab size
        token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
        df[f"MSA_model"] = df.apply(
            lambda row: label_row(
                row[mutant_col], args.sequence, token_probs.detach().cpu(), aa2tok_d, args.offset_idx
            ),
            axis=1,
        )
    # Save results
    df.to_csv(args.dms_output, index=False)

if __name__ == "__main__":
    main()