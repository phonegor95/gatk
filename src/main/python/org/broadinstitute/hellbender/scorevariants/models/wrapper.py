import torch.nn.functional as F
import pytorch_lightning as pl
from scorevariants.models.common import predictions_to_score
from torch import distributed as dist
from torch.distributed import all_gather_object, get_world_size
from natsort import natsorted

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
            self.tmp_file = open(tmp_file, 'w')

    def _dedup(self, merged_results):
        sorted_results = natsorted(merged_results, key=lambda x: (x['chrom'], x['pos'].item()))
        
        unique_results = []
        previous_result = None

        for result in sorted_results:
            if previous_result and result['pos'].item() == previous_result['pos'].item() and result['alt'] == previous_result['alt']:
                continue
            unique_results.append(result)
            previous_result = result

        return unique_results
    
    def forward(self, batch):
        outputs = self.model(batch)
        return outputs

    def on_test_end(self):
        if dist.is_initialized():
            dist.barrier()

            all_results = [None] * dist.get_world_size()
            dist.all_gather_object(all_results, self.test_results)
            merged_results = sum(all_results, [])
            unique_results = self._dedup(merged_results)    

            if self.global_rank == 0 and self.tmp_file:
                for result in unique_results:
                    self.tmp_file.write('%s\t%d\t%s\t[%s]\t%.3f\n' % (
                        result['chrom'], result['pos'], result['ref'], result['alt'], result['score']))
        
        self.tmp_file.close()
        
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
