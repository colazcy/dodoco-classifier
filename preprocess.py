# Transform Common Voice dataset into AudioFolder format

import argparse
import os
import csv
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",type=str)
parser.add_argument("--output_dir",type=str)
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir,exist_ok=True)

with open(os.path.join(args.input_dir,"validated.tsv"),"r") as f:
    dict_reader = csv.DictReader(f, delimiter="\t")
    
    for row in dict_reader:
        path = row["path"]
        shutil.copyfile(
            os.path.join(args.input_dir,"clips", path),
            os.path.join(args.output_dir, path)
        )
