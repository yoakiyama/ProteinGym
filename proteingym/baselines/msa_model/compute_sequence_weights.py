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

    # Compute fitness scores
    print("Computing MSA sequence weights...")
    args.offset_idx = msa_start_index
    # Process MSA
    processed_msa = process_msa(filename=args.msa_path, weight_filename=MSA_weight_file_name, filter_msa=args.filter_msa, hhfilter_min_cov=args.hhfilter_min_cov, hhfilter_max_seq_id=args.hhfilter_max_seq_id, hhfilter_min_seq_id=args.hhfilter_min_seq_id, path_to_hhfilter=args.path_to_hhfilter)

if __name__ == "__main__":
    main()