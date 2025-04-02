# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
import torch.autograd.gradcheck
import torch.nn.functional as F
from arctic_embed.core.losses import _MemoryEfficientInfoNCE
from arctic_embed.core.losses import info_nce_loss
from arctic_embed.core.losses import one_size_truncated_mrl_info_nce_loss
from torch.testing import assert_close


def _setup() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    Q_EMB = torch.tensor(
        [[1.0, 2.0, 3.0], [1.0, 3.0, 2.0], [2.0, 1.0, 3.0], [2.0, 3.0, 1.0]],
        requires_grad=True,
    )
    D_EMB = torch.tensor(
        [
            [1.1, 2.1, 3.1],
            [1.1, 3.1, 2.1],
            [2.1, 1.1, 3.1],
            [2.1, 3.1, 1.1],
            [2.2, 3.2, 1.2],
        ],
        requires_grad=True,
    )
    RELATIONS = torch.tensor(
        [
            [1, -1, -1, -1, -1],
            [-1, 1, -1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, -1, 1, -1],
        ]
    )

    S = torch.matmul(F.normalize(Q_EMB, dim=1), F.normalize(D_EMB, dim=1).transpose(0, 1))

    # We could just define S directly, but I think it's useful to set it up, since
    # this allows us to more easily change S by construction.
    # print([[round(x, 5) for x in row] for row in S.tolist()])
    assert_close(
        S,
        torch.tensor(
            [
                [0.99986, 0.93138, 0.93138, 0.79441, 0.80222],
                [0.93138, 0.99986, 0.79441, 0.93138, 0.93373],
                [0.93138, 0.79441, 0.99986, 0.72593, 0.73646],
                [0.79441, 0.93138, 0.72593, 0.99986, 0.99948],
            ]
        ),
    )
    return Q_EMB, D_EMB, S, RELATIONS


def test_info_nce() -> None:
    Q_EMB, D_EMB, S, RELATIONS = _setup()
    # Verify loss implementation equivalent.
    with torch.no_grad():
        want = F.cross_entropy(S / 0.1, (RELATIONS == 1).to(torch.float))
        got = info_nce_loss(S, RELATIONS, temperature=0.1)
        assert_close(actual=got, expected=want)

    # Verify loss is differentiable.
    loss = info_nce_loss(S, RELATIONS, temperature=0.1)
    loss.backward()
    assert Q_EMB.grad is not None
    assert_close(actual=Q_EMB.grad.sum(), expected=torch.tensor(0.2048036754131317))


def test_info_nce_two_positive() -> None:
    Q_EMB, D_EMB, S, RELATIONS = _setup()
    # Flip the hard negative to a positive so that the last query has two positive docs.
    relations = RELATIONS.clone()
    relations[-1, -1] = 1

    # Verify loss implementation equivalent.
    with torch.no_grad():
        want_initial = F.cross_entropy(S[:-1], (relations[:-1] == 1).to(torch.float), reduction="none")
        want_double_positive = F.cross_entropy(
            torch.row_stack([S[-1, [0, 1, 2, 3]], S[-1, [0, 1, 2, 4]]]),
            (torch.row_stack([relations[-1, [0, 1, 2, 3]], relations[-1, [0, 1, 2, 4]]]) == 1).to(torch.float),
            reduction="none",
        )
        want = torch.cat([want_initial, want_double_positive]).mean()
        got = info_nce_loss(S, relations, temperature=1.0)
    assert_close(actual=got, expected=want)

    # Verify loss is differentiable.
    loss = info_nce_loss(S, relations, temperature=1.0)
    loss.backward()
    assert Q_EMB.grad is not None
    assert_close(actual=Q_EMB.grad.sum(), expected=torch.tensor(0.04957890510559082))


def test_infonce_efficient() -> None:
    # Define a reference implementation of the loss function.
    def reference_info_nce_loss(
        scores: torch.Tensor, relations: torch.Tensor, temperature: float = 0.01
    ) -> torch.Tensor:
        # Temperature scale the scores.
        s = scores / temperature

        # Expand cases of multiple positives so that each row has just a single
        # positive.
        # (NOTE: This also drops cases with zero positives).
        # Find where the positives are.
        pos_row_idx, pos_col_idx = torch.where(relations == 1)
        # NOTE: Since this expansion requires copying tensors and consuming more
        # memory, we only do it when we have to.
        if pos_row_idx.size(0) != scores.size(0):
            # Work on a copy of the original matrix.
            relations = relations.clone()
            # Zero out positives.
            relations[pos_row_idx, pos_col_idx] = 0
            # Expand to one row per positives.
            relations = relations[pos_row_idx]
            # Un-zero-out one positive per row.
            relations[torch.arange(relations.size(0)), pos_col_idx] = 1
            # Expand the scores to one row per positives.
            s = s[pos_row_idx]

        is_positive = relations > 0
        is_unlabeled = relations == 0
        log_numerator = s[is_positive]
        log_denominator = torch.logsumexp(s.masked_fill(is_unlabeled, -1e12), dim=1)
        return -(log_numerator - log_denominator).mean()

    # Start with a basic test case.
    _Q_EMB, _D_EMB, S, RELATIONS = _setup()

    # Set the internal sub-batch size to 2 to test batching even with our small
    # test case.
    _MemoryEfficientInfoNCE.FWD_SUB_BATCH_SIZE = 2  # type: ignore

    # Specify a more complicated set of relations where the first document is relevant
    # to the first and second queries.
    RELATIONS_COMPLEX = torch.tensor(
        [
            [1, -1, -1, -1, -1],
            [1, 1, -1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, -1, 1, -1],
        ]
    )

    # Ensure loss values are the same.
    with torch.no_grad():
        want_loss = reference_info_nce_loss(S, RELATIONS_COMPLEX, temperature=0.1)
        got_loss = info_nce_loss(S, RELATIONS_COMPLEX, temperature=0.1)
        assert not torch.isclose(info_nce_loss(S, RELATIONS, temperature=0.1), want_loss)
        assert_close(actual=got_loss, expected=want_loss)

    # Check gradient using finite differencing.
    # NOTE: The finite differencing is only close enough when setting the temperature
    # and eps values to be fairly high, but this gives some confidence at least!
    torch.autograd.gradcheck(lambda s: info_nce_loss(s, RELATIONS_COMPLEX, 10.0), (S,), eps=5e-3)  # type: ignore

    # Ensure that the gradients are the same as another implementation.
    _loss, [want_grad] = torch.autograd.functional.vjp(
        lambda s: 2 * reference_info_nce_loss(s, RELATIONS_COMPLEX, 0.1), (S,)
    )
    _loss, [got_grad] = torch.autograd.functional.vjp(lambda s: 2 * info_nce_loss(s, RELATIONS_COMPLEX, 0.1), (S,))
    assert_close(actual=got_grad, expected=want_grad)


@torch.inference_mode()
def test_one_size_truncated_mrl_info_nce_loss() -> None:
    Q_EMB, D_EMB, S, RELATIONS = _setup()
    truncated_dim = 2
    S_truncated = F.normalize(Q_EMB[:, :truncated_dim], dim=1) @ F.normalize(D_EMB[:, :truncated_dim], dim=1).T
    want_full = F.cross_entropy(S / 0.1, (RELATIONS == 1).to(torch.float))
    want_truncated = F.cross_entropy(S_truncated / 0.1, (RELATIONS == 1).to(torch.float))
    want = 0.75 * want_full + 0.25 * want_truncated
    got, *_ = one_size_truncated_mrl_info_nce_loss(
        query_embeddings=Q_EMB,
        document_embeddings=D_EMB,
        relations=RELATIONS,
        truncated_dimension=truncated_dim,
        mrl_weight=0.25,
        temperature=0.1,
    )
    assert_close(actual=got, expected=want)
