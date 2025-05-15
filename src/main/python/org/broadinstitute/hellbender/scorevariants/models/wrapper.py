import torch.nn.functional as F
import pytorch_lightning as pl
from scorevariants.models.common import predictions_to_score
from torch import distributed as dist
from torch.distributed import all_gather_object, get_world_size
import sys
import os

class LightningWrapper(pl.LightningModule):
    def __init__(self, model,
                 tmp_file=None,
                 **additional_hparams,
                ):
        """PyTorch Lighting wrapper for training and evaluation
        
        Args:
            model: nn.Module, pytorch model to train/evaluate
            additional_hparams: saved in hparams
        
        """
        super().__init__()
        self.model = model
        
        # save all hyperparameters
        # in addition to provenance, this helps
        # with model checkpointing by saving the model
        self.save_hyperparameters()
        self.test_results = []
        self.tmp_file = tmp_file
    
    def forward(self, batch):
        outputs = self.model(batch)
        return outputs

    def on_test_start(self):
        if self.tmp_file:
            # Set the tmp_file after the distributed initialization
            tmp_files = f".{self.tmp_file}.{self.global_rank}.tmp"
            if os.path.exists(tmp_files):
                os.remove(tmp_files)
            self._tmp_fh = open(tmp_files, 'a')
            sys.stdout.write(f"Rank {self.global_rank} is writing to {self.tmp_file}\n")

    def on_test_end(self):
        if dist.is_initialized():
            dist.barrier()
        self.tmp_file.close()
        
    def test_step(self, batch, batch_idx):
        predictions = self(batch)
        detached_predictions = predictions.detach()
        prob_predictions = F.softmax(detached_predictions)
        scores = predictions_to_score(prob_predictions.detach(), batch['type'])
        for i, score in enumerate(scores):
            chrom = batch['chrom'][i]
            pos   = batch['pos'][i].cpu().item()
            ref   = batch['ref'][i]
            alt   = batch['alt'][i]
            self._tmp_fh.write(f"{chrom}\t{pos}\t{ref}\t[{alt}]\t{score.cpu().item():.3f}\n")
