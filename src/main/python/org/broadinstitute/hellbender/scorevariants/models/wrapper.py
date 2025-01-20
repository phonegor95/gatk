import torch.nn.functional as F
import pytorch_lightning as pl
from scorevariants.models.common import predictions_to_score
import numpy as np

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
        if tmp_file:
            self.tmp_file = tmp_file 

    def forward(self, batch):
        outputs = self.model(batch)
        return outputs

    def on_test_end(self):
        with open(self.tmp_file, "w") as f:
            for result in self.test_results:
                f.write('%s\t%d\t%s\t[%s]\t%.3f\n' % (
                    result['chrom'], result['pos'], result['ref'], result['alt'], result['score']))
        
    def test_step(self, batch, batch_idx):
        predictions = self(batch)
        detached_predictions = predictions.detach()
        prob_predictions = F.softmax(detached_predictions)
        scores = predictions_to_score(prob_predictions.detach(), batch['type'])
        for i in range(len(scores)):
            self.test_results.append({
                'chrom': batch['chrom'][i],
                'pos': batch['pos'][i],
                'ref': batch['ref'][i],
                'alt': batch['alt'][i],
                'score': scores[i]
            })
