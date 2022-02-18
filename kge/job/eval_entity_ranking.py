import math
import time
import sys
from typing import List, Tuple, Union

import torch
from torch import Tensor
import kge.job
from kge.job import EvaluationJob, Job
from kge import Config, Dataset
from collections import defaultdict


class EntityRankingJob(EvaluationJob):
    """ Entity ranking evaluation protocol """

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        self.config.check(
            "entity_ranking.tie_handling.type",
            ["rounded_mean_rank", "best_rank", "worst_rank"],
        )
        self.tie_handling = self.config.get("entity_ranking.tie_handling.type")

        self.tie_atol = float(self.config.get("entity_ranking.tie_handling.atol"))
        self.tie_rtol = float(self.config.get("entity_ranking.tie_handling.rtol"))

        self.filter_with_test = config.get("entity_ranking.filter_with_test")
        self.filter_splits = self.config.get("entity_ranking.filter_splits")
        if self.eval_split not in self.filter_splits:
            self.filter_splits.append(self.eval_split)

        self.query_types = [
            key
            for key, enabled in self.config.get("entity_ranking.query_types").items()
            if enabled
        ]
        assert len(self.query_types) > 0 # have at least one query type during evaluation 

        max_k = min(
            self.dataset.num_entities(),
            max(self.config.get("entity_ranking.hits_at_k_s")),
        )
        self.hits_at_k_s = list(
            filter(lambda x: x <= max_k, self.config.get("entity_ranking.hits_at_k_s"))
        )

        #: Whether to create additional histograms for head and tail slot
        self.head_and_tail = config.get("entity_ranking.metrics_per.head_and_tail")
        #: Whether to create additional histograms for relation prediction
        self.relation_prediction = config.get("entity_ranking.metrics_per.relation_prediction")
        if self.relation_prediction: assert 'so_to_p' in self.query_types
        #: Whether to create additional histograms for entity prediction
        self.entity_prediction = config.get("entity_ranking.metrics_per.entity_prediction")
        if self.entity_prediction: assert 'sp_to_o' in self.query_types and 'po_to_s' in self.query_types

        #: Hooks after computing the ranks for each batch entry.
        #: Signature: hists, s, p, o, s_ranks, o_ranks, job, **kwargs
        self.hist_hooks = [hist_all]
        if config.get("entity_ranking.metrics_per.relation_type"):
            self.hist_hooks.append(hist_per_relation_type)
        if config.get("entity_ranking.metrics_per.argument_frequency"):
            self.hist_hooks.append(hist_per_frequency_percentile)

        if self.__class__ == EntityRankingJob:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        super()._prepare()
        """Construct all indexes needed to run."""

        # create data and precompute indexes
        self.triples = self.dataset.split(self.config.get("eval.split"))
        for split in self.filter_splits:
            for query_type in self.query_types:
                self.dataset.index(f"{split}_{query_type}")
        if "test" not in self.filter_splits and self.filter_with_test:
            for query_type in self.query_types:
                self.dataset.index(f"test_{query_type}")

        # and data loader
        self.loader = torch.utils.data.DataLoader(
            self.triples,
            collate_fn=self._collate,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.config.get("eval.num_workers"),
            pin_memory=self.config.get("eval.pin_memory"),
        )

    def _collate(self, batch):
        "Looks up true triples for each triple in the batch"
        label_coords = []
        batch = torch.cat(batch).reshape((-1, 3))
        for split in self.filter_splits:
            split_label_coords = kge.job.util.get_coords_from_spo_batch(
                batch=batch,
                dataset=self.dataset,
                query_types=self.query_types,
                set_type=split
            )
            label_coords.append(split_label_coords)
        label_coords = torch.cat(label_coords)

        if "test" not in self.filter_splits and self.filter_with_test:
            test_label_coords = kge.job.util.get_coords_from_spo_batch(
                batch=batch,
                dataset=self.dataset,
                query_types=self.query_types,
                set_type="test"
            )
        else:
            test_label_coords = torch.zeros([0, 2], dtype=torch.long)

        return batch, label_coords, test_label_coords

    @torch.no_grad()
    def _evaluate(self):
        num_entities = self.dataset.num_entities()
        num_relations = self.dataset.num_relations()

        # number of total entries 
        num_elements = 0
        for query_typle in self.query_types:
            if query_typle == 'so_to_p':
                num_elements += num_relations 
            else:
                num_elements += num_entities

        # we also filter with test data if requested
        filter_with_test = "test" not in self.filter_splits and self.filter_with_test

        # which rankings to compute (DO NOT REORDER; code assumes the order given here)
        rankings = (
            ["_raw", "_filt", "_filt_test"] if filter_with_test else ["_raw", "_filt"]
        )

        # dictionary that maps entry of rankings to a sparse tensor containing the
        # true labels for this option
        labels_for_ranking = defaultdict(lambda: None)

        # Initiliaze dictionaries that hold the overall histogram of ranks of true
        # answers. These histograms are used to compute relevant metrics. The dictionary
        # entry with key 'all' collects the overall statistics and is the default.
        hists = dict()
        hists_filt = dict()
        hists_filt_test = dict()

        # create initial trace entry
        self.current_trace["epoch"] = dict(
            type="entity_ranking",
            scope="epoch",
            split=self.eval_split,
            filter_splits=self.filter_splits,
            epoch=self.epoch,
            batches=len(self.loader),
            size=len(self.triples),
        )

        # run pre-epoch hooks (may modify trace)
        for f in self.pre_epoch_hooks:
            f(self)

        # let's go
        epoch_time = -time.time()
        for batch_number, batch_coords in enumerate(self.loader):
            # create initial batch trace (yet incomplete)
            self.current_trace["batch"] = dict(
                type="entity_ranking",
                scope="batch",
                split=self.eval_split,
                filter_splits=self.filter_splits,
                epoch=self.epoch,
                batch=batch_number,
                size=len(batch_coords[0]),
                batches=len(self.loader),
            )

            # run the pre-batch hooks (may update the trace)
            for f in self.pre_batch_hooks:
                f(self)

            # construct a sparse label tensor of shape batch_size x 2*num_entities
            # entries are either 0 (false) or infinity (true)
            # TODO add timing information
            batch = batch_coords[0].to(self.device)
            s, p, o = batch[:, 0], batch[:, 1], batch[:, 2]
            label_coords = batch_coords[1].to(self.device)
            if filter_with_test:
                test_label_coords = batch_coords[2].to(self.device)
                # create sparse labels tensor
                test_labels = kge.job.util.coord_to_sparse_tensor(
                    len(batch),
                    num_elements,
                    test_label_coords,
                    self.device,
                    float("Inf"),
                )
                labels_for_ranking["_filt_test"] = test_labels

            # create sparse labels tensor
            labels = kge.job.util.coord_to_sparse_tensor(
                len(batch), num_elements, label_coords, self.device, float("Inf")
            )
            labels_for_ranking["_filt"] = labels

            # compute true scores beforehand, since we can't get them from a chunked
            # score table
            # o_true_scores = self.model.score_spo(s, p, o, "o").view(-1)
            # s_true_scores = self.model.score_spo(s, p, o, "s").view(-1)
            # scoring with spo vs sp and po can lead to slight differences for ties
            # due to floating point issues.
            # We use score_sp and score_po to stay consistent with scoring used for
            # further evaluation.
            true_scores = dict()
            for query_type in self.query_types:
                if 'sp_to_o' == query_type:
                    unique_o, unique_o_inverse = torch.unique(o, return_inverse=True)
                    true_scores[query_type] = torch.gather(
                        self.model.score_sp(s, p, unique_o),
                        1,
                        unique_o_inverse.view(-1, 1),
                    ).view(-1)
                if 'po_to_s' == query_type:
                    unique_s, unique_s_inverse = torch.unique(s, return_inverse=True)
                    true_scores[query_type] = torch.gather(
                        self.model.score_po(p, o, unique_s),
                        1,
                        unique_s_inverse.view(-1, 1),
                    ).view(-1)
                if 'so_to_p' == query_type:
                    unique_p, unique_p_inverse = torch.unique(p, return_inverse=True)
                    true_scores[query_type] = torch.gather(
                        self.model.score_so(s, o, unique_p),
                        1,
                        unique_p_inverse.view(-1, 1),
                    ).view(-1)

            # default dictionary storing rank and num_ties for each key in rankings
            # as list of len 2: [rank, num_ties]
            ranks_and_ties_for_ranking = defaultdict(
                lambda: [
                    torch.zeros(s.size(0), dtype=torch.long, device=self.device),
                    torch.zeros(s.size(0), dtype=torch.long, device=self.device),
                ]
            )

            # calculate scores in chunks to not have the complete score matrix in memory
            # a chunk here represents a range of entity_values to score against
            if self.config.get("entity_ranking.chunk_size") > -1:
                entity_chunk_size = self.config.get("entity_ranking.chunk_size")
            else:
                entity_chunk_size = self.dataset.num_entities()

            if 'sp_to_o' in self.query_types or 'po_to_s' in self.query_types:
                num_chunks = math.ceil(num_entities / entity_chunk_size)
            else:
                num_chunks = math.ceil(num_relations / entity_chunk_size)

            relation_chunk_size = math.ceil(num_relations / num_chunks) if 'so_to_p' in self.query_types else 0
            
            # process chunk by chunk
            for chunk_number in range(num_chunks):
                entity_chunk_start = entity_chunk_size * chunk_number
                entity_chunk_end = min(entity_chunk_size * (chunk_number + 1), num_entities)
                relation_chunk_start = relation_chunk_size * chunk_number
                relation_chunk_end = min(relation_chunk_size * (chunk_number + 1), num_relations)

                # compute scores of chunk
                (
                    scores,
                    e_in_chunk_mask,
                    e_in_chunk
                ) = self._get_score(
                    batch, true_scores, 
                    entity_chunk_start, entity_chunk_end,
                    relation_chunk_start, relation_chunk_end
                )

                # scores = self.model.score_sp_po(
                #     s, p, o, torch.arange(entity_chunk_start, entity_chunk_end, device=self.device)
                # )
                # scores_sp = scores['sp_to_o']
                # scores_po = scores['po_to_s']
                # replace the precomputed true_scores with the ones occurring in the
                # scores matrix to avoid floating point issues
                # s_in_chunk_mask = (entity_chunk_start <= s) & (s < entity_chunk_end)
                # o_in_chunk_mask = (entity_chunk_start <= o) & (o < entity_chunk_end)
                # o_in_chunk = (o[o_in_chunk_mask] - entity_chunk_start).long()
                # s_in_chunk = (s[s_in_chunk_mask] - entity_chunk_start).long()

                # assert torch.equal(s_in_chunk_mask, e_in_chunk_mask['po_to_s'])
                # assert torch.equal(o_in_chunk_mask, e_in_chunk_mask['sp_to_o'])
                # assert torch.equal(o_in_chunk, e_in_chunk['sp_to_o'])
                # assert torch.equal(s_in_chunk, e_in_chunk['po_to_s'])

                # check that scoring is consistent up to configured tolerance
                # if this is not the case, evaluation metrics may be artificially inflated
                close_check = True
                for query_type in self.query_types:
                    close_check = torch.allclose(
                        scores[query_type][e_in_chunk_mask[query_type], e_in_chunk[query_type]],
                        true_scores[query_type][e_in_chunk_mask[query_type]],
                        rtol=self.tie_rtol,
                        atol=self.tie_atol,
                    )
                if not close_check:
                    diff_all = dict()
                    for query_type in self.query_types:
                        diff_all[query_type] = torch.abs(
                            scores[query_type][e_in_chunk_mask[query_type], e_in_chunk[query_type]]
                            - true_scores[query_type][e_in_chunk_mask[query_type]]
                        )
                    print(tuple(diff_all.values()))
                    diff_all = torch.cat(tuple(diff_all.values()))
                    self.config.log(
                        f"Tie-handling: mean difference between scores was: {diff_all.mean()}."
                    )
                    self.config.log(
                        f"Tie-handling: max difference between scores was: {diff_all.max()}."
                    )
                    error_message = "Error in tie-handling. The scores assigned to a triple by the SPO and SP_/_PO scoring implementations were not 'equal' given the configured tolerances. Verify the model's scoring implementations or consider increasing tie-handling tolerances."
                    if self.config.get("entity_ranking.tie_handling.warn_only"):
                        print(error_message, file=sys.stderr)
                    else:
                        raise ValueError(error_message)

                # now compute the rankings (assumes order: None, _filt, _filt_test)
                for ranking in rankings:
                    if labels_for_ranking[ranking] is None:
                        labels_chunk = None
                    else:
                        # densify the needed part of the sparse labels tensor
                        labels_chunk = self._densify_chunk_of_labels(
                            labels_for_ranking[ranking],
                            entity_chunk_start, entity_chunk_end,
                            relation_chunk_start, relation_chunk_end,
                        )

                        # remove current example from labels
                        labels_chunk[e_in_chunk_mask['sp_to_o'], e_in_chunk['sp_to_o']] = 0
                        labels_chunk[
                            e_in_chunk_mask['po_to_s'], e_in_chunk['po_to_s'] + (entity_chunk_end - entity_chunk_start)
                        ] = 0

                    # compute partial ranking and filter the scores (sets scores of true
                    # labels to infinity)
                    (
                        rank_chunk,
                        num_ties_chunk,
                        scores_filt,
                    ) = self._filter_and_rank(
                        scores, labels_chunk, true_scores
                    )

                    # from now on, use filtered scores
                    scores = scores_filt

                    # update rankings
                    for query_type in self.query_types:
                        pred_e = query_type[-1]
                        ranks_and_ties_for_ranking[pred_e + ranking][0] += rank_chunk[query_type]
                        ranks_and_ties_for_ranking[pred_e + ranking][1] += num_ties_chunk[query_type]

                # we are done with the chunk

            # We are done with all chunks; calculate final ranks from counts
            ranks = dict()
            ranks_filt = dict()
            for query_type in self.query_types:
                pred_e = query_type[-1]
                ranks[query_type] = self._get_ranks(
                    ranks_and_ties_for_ranking[f"{pred_e}_raw"][0],
                    ranks_and_ties_for_ranking[f"{pred_e}_raw"][1],
                )
                ranks_filt[query_type] = self._get_ranks(
                    ranks_and_ties_for_ranking[f"{pred_e}_filt"][0],
                    ranks_and_ties_for_ranking[f"{pred_e}_filt"][1],
                )
            # s_ranks = self._get_ranks(
            #     ranks_and_ties_for_ranking["s_raw"][0],
            #     ranks_and_ties_for_ranking["s_raw"][1],
            # )
            # o_ranks = self._get_ranks(
            #     ranks_and_ties_for_ranking["o_raw"][0],
            #     ranks_and_ties_for_ranking["o_raw"][1],
            # )
            # s_ranks_filt = self._get_ranks(
            #     ranks_and_ties_for_ranking["s_filt"][0],
            #     ranks_and_ties_for_ranking["s_filt"][1],
            # )
            # o_ranks_filt = self._get_ranks(
            #     ranks_and_ties_for_ranking["o_filt"][0],
            #     ranks_and_ties_for_ranking["o_filt"][1],
            # )

            # assert torch.equal(o_ranks, ranks['sp_to_o'])
            # assert torch.equal(s_ranks, ranks['po_to_s'])
            # assert torch.equal(o_ranks_filt, ranks_filt['sp_to_o'])
            # assert torch.equal(s_ranks_filt, ranks_filt['po_to_s'])


            # Update the histograms of of raw ranks and filtered ranks
            batch_hists = dict()
            batch_hists_filt = dict()
            for f in self.hist_hooks:
                f(batch_hists, s, p, o, ranks, job=self)
                f(batch_hists_filt, s, p, o, ranks_filt, job=self)

            # and the same for filtered_with_test ranks
            if filter_with_test:
                ranks_filt_test = dict()
                batch_hists_filt_test = dict()
                for query_type in self.query_types:
                    pred_entity = query_type[-1]
                    ranks_filt_test[query_type] = self._get_ranks(
                        ranks_and_ties_for_ranking[f"{pred_entity}_filt_test"][0],
                        ranks_and_ties_for_ranking[f"{pred_entity}_filt_test"][1],
                    )
                for f in self.hist_hooks:
                    f(
                        batch_hists_filt_test,
                        s,
                        p,
                        o,
                        ranks_filt_test,
                        job=self,
                    )

            # optionally: trace ranks of each example
            if self.trace_examples:
                entry = {
                    "type": "entity_ranking",
                    "scope": "example",
                    "split": self.eval_split,
                    "filter_splits": self.filter_splits,
                    "size": len(batch),
                    "batches": len(self.loader),
                    "epoch": self.epoch,
                }
                for i in range(len(batch)):
                    entry["batch"] = i
                    entry["s"], entry["p"], entry["o"] = (
                        s[i].item(),
                        p[i].item(),
                        o[i].item(),
                    )
                    for query_type in self.query_types:
                        if filter_with_test:
                            entry["rank_filtered_with_test"] = (
                                ranks_filt_test[query_type][i].item() + 1
                            )
                            self.trace(
                                event="example_rank",
                                task=query_type[:1],
                                rank=ranks[query_type][i].item() + 1,
                                rank_filtered=ranks_filt[query_type][i].item() + 1,
                                **entry,
                            )

            # Compute the batch metrics for the full histogram (key "all")
            metrics = self._compute_metrics(batch_hists["all"])
            metrics.update(
                self._compute_metrics(batch_hists_filt["all"], suffix="_filtered")
            )
            if filter_with_test:
                metrics.update(
                    self._compute_metrics(
                        batch_hists_filt_test["all"], suffix="_filtered_with_test"
                    )
                )

            # update batch trace with the results
            self.current_trace["batch"].update(metrics)

            # run the post-batch hooks (may modify the trace)
            for f in self.post_batch_hooks:
                f(self)

            # output, then clear trace
            if self.trace_batch:
                self.trace(**self.current_trace["batch"])
            self.current_trace["batch"] = None

            # output batch information to console
            self.config.print(
                (
                    "\r"  # go back
                    + "{}  batch:{: "
                    + str(1 + int(math.ceil(math.log10(len(self.loader)))))
                    + "d}/{}, mrr (filt.): {:4.3f} ({:4.3f}), "
                    + "hits@1: {:4.3f} ({:4.3f}), "
                    + "hits@{}: {:4.3f} ({:4.3f})"
                    + "\033[K"  # clear to right
                ).format(
                    self.config.log_prefix,
                    batch_number,
                    len(self.loader) - 1,
                    metrics["mean_reciprocal_rank"],
                    metrics["mean_reciprocal_rank_filtered"],
                    metrics["hits_at_1"],
                    metrics["hits_at_1_filtered"],
                    self.hits_at_k_s[-1],
                    metrics["hits_at_{}".format(self.hits_at_k_s[-1])],
                    metrics["hits_at_{}_filtered".format(self.hits_at_k_s[-1])],
                ),
                end="",
                flush=True,
            )

            # merge batch histograms into global histograms
            def merge_hist(target_hists, source_hists):
                for key, hist in source_hists.items():
                    if key in target_hists:
                        target_hists[key] = target_hists[key] + hist
                    else:
                        target_hists[key] = hist

            merge_hist(hists, batch_hists)
            merge_hist(hists_filt, batch_hists_filt)
            if filter_with_test:
                merge_hist(hists_filt_test, batch_hists_filt_test)

        # we are done; compute final metrics
        self.config.print("\033[2K\r", end="", flush=True)  # clear line and go back
        for key, hist in hists.items():
            name = "_" + key if key != "all" else ""
            metrics.update(self._compute_metrics(hists[key], suffix=name))
            metrics.update(
                self._compute_metrics(hists_filt[key], suffix="_filtered" + name)
            )
            if filter_with_test:
                metrics.update(
                    self._compute_metrics(
                        hists_filt_test[key], suffix="_filtered_with_test" + name
                    )
                )
        epoch_time += time.time()

        # update trace with results
        self.current_trace["epoch"].update(
            dict(epoch_time=epoch_time, event="eval_completed", **metrics,)
        )

    def _densify_chunk_of_labels(
        self, labels: torch.Tensor, 
        entity_chunk_start: int, entity_chunk_end: int,
        relation_chunk_start: int, relation_chunk_end: int,
    ) -> torch.Tensor:
        """Creates a dense chunk of a sparse label tensor.

        A chunk here is a range of entity values with 'chunk_start' being the lower
        bound and 'chunk_end' the upper bound.

        The resulting tensor contains the labels for the sp chunk and the po chunk.

        :param labels: sparse tensor containing the labels corresponding to the batch
        for sp and po

        :param chunk_start: int start index of the chunk

        :param chunk_end: int end index of the chunk

        :return: batch_size x chunk_size*2 dense tensor with labels for the sp chunk and
        the po chunk.

        """
        # intialize
        num_entities = self.dataset.num_entities()
        num_relations = self.dataset.num_relations()
        indices = labels._indices()
        indices_chunk = {}
        mask = {}
        previous_chunk_size = 0
        previous_elements = 0
        start_point = 0
        total_chunk_size = 0
        
        #go
        for query_type in self.query_types:
            chunk_start = relation_chunk_start if query_type == 'so_to_p' else entity_chunk_start
            chunk_end = relation_chunk_end if query_type == 'so_to_p' else entity_chunk_end

            mask[query_type] = (start_point + chunk_start <= indices[1, :]) & (indices[1, :] < start_point + chunk_end)
            indices_chunk[query_type] = indices[:, mask[query_type]]
            indices_chunk[query_type][1, :] = (
                indices_chunk[query_type][1, :] - chunk_start - previous_elements + previous_chunk_size 
                )

            previous_elements += num_relations if query_type == 'so_to_p' else num_entities
            start_point += previous_elements
            previous_chunk_size = chunk_end - chunk_start
            total_chunk_size += previous_chunk_size

        final_mask = mask[self.query_types[0]]
        for query_type in self.query_types:
            final_mask |= mask[query_type]

        indices_chunk = torch.cat(tuple(indices_chunk.values()), dim=1)
        dense_labels = torch.sparse.LongTensor(
            indices_chunk,
            # since all sparse label tensors have the same value we could also
            # create a new tensor here without indexing with:
            # torch.full([indices_chunk.shape[1]], float("inf"), device=self.device)
            labels._values()[final_mask],
            torch.Size([labels.size()[0], total_chunk_size]),
        ).to_dense()
        return dense_labels

        # mask_sp = (entity_chunk_start <= indices[1, :]) & (indices[1, :] < entity_chunk_end)
        # mask_po = ((entity_chunk_start + num_entities) <= indices[1, :]) & (
        #     indices[1, :] < (entity_chunk_end + num_entities)
        # )
        # indices_sp_chunk = indices[:, mask_sp]
        # indices_sp_chunk[1, :] = indices_sp_chunk[1, :] - entity_chunk_start

        # indices_po_chunk = indices[:, mask_po]
        # indices_po_chunk[1, :] = (
        #     indices_po_chunk[1, :] - num_entities - entity_chunk_start * 2 + entity_chunk_end
        # )

        # assert torch.equal(mask['sp_to_o'], mask_sp)
        # assert torch.equal(mask['po_to_s'], mask_po)
        # assert torch.equal(indices_chunk['sp_to_o'], indices_sp_chunk)
        # assert torch.equal(indices_chunk['sp_to_o'], indices_sp_chunk)

    def _filter_and_rank(
        self,
        scores: dict,
        labels: torch.Tensor,
        true_scores: dict
    ):
        """Filters the current examples with the given labels and returns counts rank and
num_ties for each true score.

        :param scores_sp: batch_size x chunk_size tensor of scores

        :param scores_po: batch_size x chunk_size tensor of scores

        :param labels: batch_size x 2*chunk_size tensor of scores

        :param o_true_scores: batch_size x 1 tensor containing the scores of the actual
        objects in batch

        :param s_true_scores: batch_size x 1 tensor containing the scores of the actual
        subjects in batch

        :return: batch_size x 1 tensors rank and num_ties for s and o and filtered
        scores_sp and scores_po

        """
        previous_size = 0
        rank = {}
        num_ties = {}
        for query_type in self.query_types:
            # remove current example from labels
            if labels is not None:
                chunk_size = scores[query_type].shape[1]
                labels_query = labels[:, previous_size : previous_size + chunk_size]
                scores[query_type] = scores[query_type] - labels_query
                previous_size += chunk_size
            rank[query_type], num_ties[query_type] = self._get_ranks_and_num_ties(scores[query_type], true_scores[query_type])
        return rank, num_ties, scores

    def _get_ranks_and_num_ties(
        self, scores: torch.Tensor, true_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns rank and number of ties of each true score in scores.

        :param scores: batch_size x entities tensor of scores

        :param true_scores: batch_size x 1 tensor containing the actual scores of the batch

        :return: batch_size x 1 tensors rank and num_ties
        """
        # process NaN values
        scores = scores.clone()
        scores[torch.isnan(scores)] = float("-Inf")
        true_scores = true_scores.clone()
        true_scores[torch.isnan(true_scores)] = float("-Inf")

        # Determine how many scores are greater than / equal to each true answer (in its
        # corresponding row of scores)
        is_close = torch.isclose(
            scores, true_scores.view(-1, 1), rtol=self.tie_rtol, atol=self.tie_atol
        )
        is_greater = scores > true_scores.view(-1, 1)
        num_ties = torch.sum(is_close, dim=1, dtype=torch.long)
        rank = torch.sum(is_greater & ~is_close, dim=1, dtype=torch.long)
        return rank, num_ties

    def _get_ranks(self, rank: torch.Tensor, num_ties: torch.Tensor) -> torch.Tensor:
        """Calculates the final rank from (minimum) rank and number of ties.

        :param rank: batch_size x 1 tensor with number of scores greater than the one of
        the true score

        :param num_ties: batch_size x tensor with number of scores equal as the one of
        the true score

        :return: batch_size x 1 tensor of ranks

        """

        if self.tie_handling == "rounded_mean_rank":
            return rank + num_ties // 2
        elif self.tie_handling == "best_rank":
            return rank
        elif self.tie_handling == "worst_rank":
            return rank + num_ties - 1
        else:
            raise NotImplementedError

    def _compute_metrics(self, rank_hist, suffix=""):
        """Computes desired matrix from rank histogram"""
        metrics = {}
        n = torch.sum(rank_hist).item()

        ranks = torch.arange(1, self.dataset.num_entities() + 1).float().to(self.device)
        metrics["mean_rank" + suffix] = (
            (torch.sum(rank_hist * ranks).item() / n) if n > 0.0 else 0.0
        )

        reciprocal_ranks = 1.0 / ranks
        metrics["mean_reciprocal_rank" + suffix] = (
            (torch.sum(rank_hist * reciprocal_ranks).item() / n) if n > 0.0 else 0.0
        )

        hits_at_k = (
            (
                torch.cumsum(
                    rank_hist[: max(self.hits_at_k_s)], dim=0, dtype=torch.float64
                )
                / n
            ).tolist()
            if n > 0.0
            else [0.0] * max(self.hits_at_k_s)
        )

        for i, k in enumerate(self.hits_at_k_s):
            metrics["hits_at_{}{}".format(k, suffix)] = hits_at_k[k - 1]

        return metrics

    def _get_score(
        self, 
        batch: Union[Tensor, List[Tensor]], true_scores: dict, 
        entity_chunk_start: int, entity_chunk_end: int,
        relation_chunk_start: int, relation_chunk_end: int
    ):
        scores = dict()
        element_in_chunk_mask = dict()
        element_in_chunk = dict()
        s, p, o = batch[:, 0], batch[:, 1], batch[:, 2]
        for query_type in self.query_types:
            if query_type == 'sp_to_o':
                pred_entity = o
                entity_subset = torch.arange(entity_chunk_start, entity_chunk_end, device=self.device)
                scores[query_type] = self.model.score_sp(
                    s, p, entity_subset
                )
                chunk_start = entity_chunk_start
                chunk_end = entity_chunk_end
            if query_type == 'po_to_s':
                pred_entity = s
                entity_subset = torch.arange(entity_chunk_start, entity_chunk_end, device=self.device)
                scores[query_type] = self.model.score_po(
                    p, o, entity_subset 
                )
                chunk_start = entity_chunk_start
                chunk_end = entity_chunk_end
            if query_type == 'so_to_p':
                pred_entity = p
                relation_subset = torch.arange(relation_chunk_start, relation_chunk_end, device=self.device)
                scores[query_type] = self.model.score_so(
                    s, o, relation_subset
                )
                chunk_start = relation_chunk_start
                chunk_end = relation_chunk_end
            # replace the precomputed true_scores with the ones occurring in the
            # scores matrix to avoid floating point issues
            element_in_chunk_mask[query_type] = (chunk_start <= pred_entity) & (pred_entity < chunk_end)
            element_in_chunk[query_type] = (
                pred_entity[element_in_chunk_mask[query_type]] - chunk_start
            ).long()
            scores[query_type][element_in_chunk_mask[query_type], element_in_chunk[query_type]] = true_scores[query_type][element_in_chunk_mask[query_type]]
        return scores, element_in_chunk_mask, element_in_chunk



# HISTOGRAM COMPUTATION ###############################################################


def __initialize_hist(hists, key, job):
    """If there is no histogram with given `key` in `hists`, add an empty one."""
    if key not in hists:
        hists[key] = torch.zeros(
            [job.dataset.num_entities()],
            device=job.config.get("job.device"),
            dtype=torch.float,
        )


def hist_all(hists, s, p, o, ranks, job, **kwargs):
    """Create histogram of all subject/object ranks (key: "all").

    `hists` a dictionary of histograms to update; only key "all" will be affected. `s`,
    `p`, `o` are true triples indexes for the batch. `s_ranks` and `o_ranks` are the
    rank of the true answer for (?,p,o) and (s,p,?) obtained from a model.

    """
    __initialize_hist(hists, "all", job)
    if job.head_and_tail:
        __initialize_hist(hists, "head", job)
        __initialize_hist(hists, "tail", job)
        hist_head = hists["head"]
        hist_tail = hists["tail"]

    hist = hists["all"]
    for query_type in ranks.keys():
        ranks_unique, ranks_count = torch.unique(ranks[query_type], return_counts=True)
        hist.index_add_(0, ranks_unique, ranks_count.float())
    
    if job.relation_prediction:
        __initialize_hist(hists, "relation", job)
        hist_relation = hists['relation']
        ranks_unique, ranks_count = torch.unique(ranks['so_to_p'], return_counts=True)
        hist_relation.index_add_(0, ranks_unique, ranks_count.float())

    if job.entity_prediction:
        __initialize_hist(hists, "entity", job)
        hist_entity = hists['entity']
        o_ranks_unique, o_ranks_count = torch.unique(ranks['sp_to_o'], return_counts=True)
        s_ranks_unique, s_ranks_count = torch.unique(ranks['po_to_s'], return_counts=True)
        hist_entity.index_add_(0, o_ranks_unique, o_ranks_count.float())
        hist_entity.index_add_(0, s_ranks_unique, s_ranks_count.float())

    if job.head_and_tail:
        ranks_unique, ranks_count = torch.unique(ranks['sp_to_o'], return_counts=True)
        hist_tail.index_add_(0, ranks_unique, ranks_count.float())
        ranks_unique, ranks_count = torch.unique(ranks['po_to_s'], return_counts=True)
        hist_head.index_add_(0, ranks_unique, ranks_count.float())


def hist_per_relation_type(hists, s, p, o, ranks, job, **kwargs):
    for rel_type, rels in job.dataset.index("relations_per_type").items():
        __initialize_hist(hists, rel_type, job)
        hist = hists[rel_type]
        if job.head_and_tail:
            __initialize_hist(hists, f"{rel_type}_head", job)
            __initialize_hist(hists, f"{rel_type}_tail", job)
            hist_head = hists[f"{rel_type}_head"]
            hist_tail = hists[f"{rel_type}_tail"]
        if job.relation_prediction:
             __initialize_hist(hists, f"{rel_type}_relation", job)
             hist_relation = hists[f"{rel_type}_relation"]

        o_ranks = ranks['sp_to_o']
        s_ranks = ranks['po_to_s']

        mask = [_p in rels for _p in p.tolist()]
        for r, m in zip(o_ranks, mask):
            if m:
                hists[rel_type][r] += 1
                if job.head_and_tail:
                    hist_tail[r] += 1

        for r, m in zip(s_ranks, mask):
            if m:
                hists[rel_type][r] += 1
                if job.head_and_tail:
                    hist_head[r] += 1

        if 'so_to_p' in ranks.keys():
            p_ranks = ranks['so_to_p']
            for r, m in zip(p_ranks, mask):
                if m:
                    hists[rel_type][r] += 1
                    if job.relation_prediction:
                        hist_relation[r] += 1


def hist_per_frequency_percentile(hists, s, p, o, ranks, job, **kwargs):
    # initialize
    frequency_percs = job.dataset.index("frequency_percentiles")
    for arg, percs in frequency_percs.items():
        for perc, value in percs.items():
            __initialize_hist(hists, "{}_{}".format(arg, perc), job)
    o_ranks = ranks['sp_to_o']
    s_ranks = ranks['po_to_s']
    # go
    for perc in frequency_percs["subject"].keys():  # same for relation and object
        for r, m_s, m_r in zip(
            s_ranks,
            [id in frequency_percs["subject"][perc] for id in s.tolist()],
            [id in frequency_percs["relation"][perc] for id in p.tolist()],
        ):
            if m_s:
                hists["{}_{}".format("subject", perc)][r] += 1
            if m_r:
                hists["{}_{}".format("relation", perc)][r] += 1
        for r, m_o, m_r in zip(
            o_ranks,
            [id in frequency_percs["object"][perc] for id in o.tolist()],
            [id in frequency_percs["relation"][perc] for id in p.tolist()],
        ):
            if m_o:
                hists["{}_{}".format("object", perc)][r] += 1
            if m_r:
                hists["{}_{}".format("relation", perc)][r] += 1
