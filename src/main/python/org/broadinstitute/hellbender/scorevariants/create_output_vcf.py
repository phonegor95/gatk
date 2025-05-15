#!/usr/bin/python3

from cyvcf2 import VCF, Writer
import re
import sys

CONTIG_INDEX = 0;
POS_INDEX = 1;
REF_INDEX = 2;
ALT_INDEX = 3;
KEY_INDEX = 4;

def create_output_vcf(vcf_in, scores_file, vcf_out, label, n_gpus):
    variant_file = VCF(vcf_in)

    variant_file.add_info_to_header({'ID': label, 'Number': 1, 'Type': 'Float', 'Description': 'Log odds of being a true variant versus being false under the trained Convolutional Neural Network'})
    w = Writer(vcf_out, variant_file)

    score_handles = [open(score_file, 'r') for score_file in [f".{scores_file}.{i}.tmp" for i in range(n_gpus)]]
    endofVCF = False

    while True:
        scoredVariants = [handle.readline() for handle in score_handles]
        for i, sv in enumerate(scoredVariants):
            try:
                variant = next(variant_file)
            except StopIteration:
                endofVCF = True
                break
            scoredVariant = sv.split('\t')
            if variant.CHROM == scoredVariant[CONTIG_INDEX] and \
               variant.POS == int(scoredVariant[POS_INDEX]) and \
               variant.REF == scoredVariant[REF_INDEX] and \
               ', '.join(variant.ALT or []) == re.sub('[\[\]]', '', scoredVariant[ALT_INDEX]):

                    if len(scoredVariant) > KEY_INDEX:
                        variant.INFO["CNN_2D"] = float(scoredVariant[KEY_INDEX])

                    w.write_record(variant)

            else:
                sys.exit(f"Score file {i} out of sync:\n"
                         f"  Score: {sv}\n"
                         f"  VCF:   {variant}\n")
        if endofVCF:
            break

    for h in score_handles:
        h.close()
    w.close()
