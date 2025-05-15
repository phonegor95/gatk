#!/usr/bin/python3

import argparse
import torch
import sys
import pytorch_lightning as pl
from lightning_fabric.utilities import seed
from torch.utils.data import DataLoader, DistributedSampler
from scorevariants.dataset import ReferenceDataset
from scorevariants.readers import TensorReader, ReferenceTensorReader
from scorevariants.models.wrapper import LightningWrapper
from scorevariants.create_output_vcf import create_output_vcf
import torch.distributed as dist
import warnings
warnings.filterwarnings("ignore")

def init_distributed_mode():
    dist.init_process_group(backend='nccl')  # 'nccl' is recommended for multi-GPU setups
    torch.cuda.set_device(dist.get_rank())  # Set the GPU device for each rank

def get_model(args, model_file):
    """
    Args:
        args: model specific and trainer arguments
        model_file: model file to load parameters from a pretrained PyTorch model
    """
    dict_args = vars(args)
    model = torch.load(model_file)
    model = LightningWrapper(model, **dict_args)
    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vcf-file', required=True, help='A VCF file containing variants to creat datasets')
    parser.add_argument('--ref-file', required=True, help='Reference sequence file to creat datasets')
    parser.add_argument('--input-file', help='BAM file containing reads to creat datasets')
    parser.add_argument('--tensor-type', default='reference', help='Name of the tensors to generate, reference for 1D reference tensors and read_tensor for 2D tensors.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=724, help='Seed to initialize the random number generator')
    parser.add_argument('--tmp-file', default='tmp.txt', help='The temporary VCF-like file where variants scores will be written')
    parser.add_argument('--output-file', required=True, help='Output VCF file')
    parser.add_argument('--accelerator', default='auto', help='Type of hardware accelerator to use (auto, cpu, cuda, mps, tpu, etc)')

    parser.add_argument('--model-directory', default='models', help='Directory containing model files')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    init_distributed_mode()
    torch.manual_seed(args.seed)
    seed.seed_everything(args.seed)
    torch.set_float32_matmul_precision('high')
    if args.tensor_type == 'reference':
        model_file = args.model_directory + '/1d_cnn_mix_train_full_bn.pt'
        tensor_reader = ReferenceTensorReader.from_files(args.vcf_file, args.ref_file)
        label = 'CNN_1D'
    elif args.tensor_type == 'read_tensor':
        model_file = args.model_directory + '/small_2d.pt'
        tensor_reader = TensorReader.from_files(args.vcf_file, args.ref_file, args.input_file)
        label = 'CNN_2D'
    else:
        sys.exit('Unknown tensor type!')
    model = get_model(args, model_file)
    test_dataset = ReferenceDataset(tensor_reader)
    
    sampler = DistributedSampler(test_dataset, shuffle=False)  # Disable shuffling to preserve order
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=11, sampler=sampler)
    n_gpus = torch.cuda.device_count()
    trainer = pl.Trainer(gradient_clip_val=1.0, accelerator=args.accelerator, strategy="ddp", devices=n_gpus, num_nodes=1)
    trainer.test(model, test_loader)
    create_output_vcf(args.vcf_file, args.tmp_file, args.output_file, label, n_gpus)

if __name__ == '__main__':
    main()
