import time

import torch
import torch.utils.data

from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn
from kge.model.reciprocal_relations_model import ReciprocalRelationsModel


class TrainingJob1vsAll(TrainingJob):
    """Samples SPO pairs and queries sp_ and _po, treating all other entities as negative."""

    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        config.log("Initializing spo training job...")
        self.type_str = "1vsAll"
        

        if self.__class__ == TrainingJob1vsAll:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()

        # determine enabled query types
        self.query_types = [
            key
            for key, enabled in self.config.get("1vsAll.query_types").items()
            if enabled
        ]
        
        # determine query weights
        self.query_weight = self.config.get("1vsAll.query_weight")

        self.num_examples = self.dataset.split(self.train_split).size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=lambda batch: {
                "triples": self.dataset.split(self.train_split)[batch, :].long()
            },
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        result.size = len(batch["triples"])

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: TrainingJob._ProcessBatchResult,
    ):
        batch_size = result.size

        # prepare
        result.prepare_time -= time.time()
        triples = batch["triples"][subbatch_slice].to(self.device)
        result.prepare_time += time.time()

        # forward/backward pass (sp)
        if 'sp_' in self.query_types:

            result.forward_time -= time.time()
            scores_sp = self.model.score_sp(triples[:, 0], triples[:, 1])
            loss_value_sp = self.loss(scores_sp, triples[:, 2]) / batch_size
            # Apply query weight
            weight = self.query_weight['sp_']
            loss_value_sp = weight*loss_value_sp
            result.avg_loss += loss_value_sp.item()
            result.forward_time += time.time()
            result.backward_time = -time.time()
            if not self.is_forward_only:
                loss_value_sp.backward()
            result.backward_time += time.time()

        # forward/backward pass (po)
        if '_po' in self.query_types:
            result.forward_time -= time.time()
            scores_po = self.model.score_po(triples[:, 1], triples[:, 2])
            loss_value_po = self.loss(scores_po, triples[:, 0]) / batch_size
            # Apply query weight
            weight = self.query_weight['_po']
            loss_value_po = weight*loss_value_po
            result.avg_loss += loss_value_po.item()
            result.forward_time += time.time()
            result.backward_time -= time.time()
            if not self.is_forward_only:
                loss_value_po.backward()
            result.backward_time += time.time()

        # forward/backward pass (so)
        if 's_o' in self.query_types:
            result.forward_time -= time.time()
            scores_so = self.model.score_so(triples[:, 0], triples[:, 2])
            loss_value_so = self.loss(scores_so, triples[:, 1]) / batch_size
            # Apply query weight
            weight = self.query_weight['s_o']
            loss_value_so = weight*loss_value_so
            result.avg_loss += loss_value_so.item()
            result.forward_time += time.time()
            result.backward_time -= time.time()
            if not self.is_forward_only:
                loss_value_so.backward()
            result.backward_time += time.time()

            ## forward/backward pass (os)
            if isinstance(self.model, ReciprocalRelationsModel):
                result.forward_time -= time.time()
                scores_os = self.model.score_so(triples[:, 2], triples[:, 0])
                loss_value_os = self.loss(scores_os, triples[:, 1] + self.dataset.num_relations()) / batch_size
                # Apply query weight
                weight = self.query_weight['s_o']
                loss_value_os = weight*loss_value_os
                result.avg_loss += loss_value_os.item()
                result.forward_time += time.time()
                result.backward_time -= time.time()
                if not self.is_forward_only:
                    loss_value_os.backward()
                result.backward_time += time.time()