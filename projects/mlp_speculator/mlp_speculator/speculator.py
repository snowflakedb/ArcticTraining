import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from .configs import MLPSpeculatorConfig


class LayerNormParameterized(nn.Module):
    """
    A generalized LayerNorm implementation. With all optional arguments set to True, equivalent to nn.LayerNorm up to epsilon stabilization term
    (this class divides inputs by min(norm, eps), while nn.LayerNorm divides by norm + eps).
    ...
    Args
    ----
    normalized_shape : int
        Dimensionality of input data (size of final tensor axis)
    eps : float
        Safety term to prevent division by zero. Make sure the chosen value fits in the range of your encoding scheme (i.e. fp16 requires eps >= 6e-8).
    elementwise_scale : bool
        Include a learned scaling term after normalization?
    elementwise_shift : bool
        Include a learned bias term after normalization?
    use_mean : bool
        Recenter inputs around zero before normalizing, or just rescale?
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
        elementwise_scale=True,
        elementwise_shift=False,
        use_mean=False,
        use_high_precision_pow=False,
    ):
        super(LayerNormParameterized, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_scale = elementwise_scale
        self.elementwise_shift = elementwise_shift
        self.use_mean = use_mean
        self.use_high_precision_pow = use_high_precision_pow

        if self.elementwise_scale:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        # else:
        #     self.register_parameter("weight", None)
        if self.elementwise_shift:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        # else:
        #     self.register_parameter("bias", None)

    def reset_parameters(self):
        if self.elementwise_scale:
            self.weight.data.fill_(1)
        if self.elementwise_shift:
            self.bias.data.zero_()

    def forward(self, x):
        if self.use_mean:
            x = x - x.mean(-1, keepdim=True)
        # x = F.normalize(x, dim=-1)*math.sqrt(x.size(-1))
        xf = x
        if self.use_high_precision_pow:
            xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        x = xf.type_as(x)
        if self.elementwise_scale:
            x = self.weight * x
        if self.elementwise_shift:
            x = x + self.bias
        return x


class MLPSpeculator(nn.Module):
    """
    This is a simple MLP-based speculator that functions similarly to Medusa
    (https://arxiv.org/abs/2401.10774), ingesting context via the final embedding
    vector from the base model. However, this model also conditions on previously
    predicted tokens, similarly to an RNN, allowing it to generate better-quality n-grams.

    The architecture is as flat and simple as possible: for each prediction head,
    the current state vector is projected into a new latent space and added to the
    previous token's embedding. This sum goes through layernorm and activation, forming
    the new state vector. This state predicts the next token (or set of candidate tokens)
    for the current head, and then is passed on to the next.
    ...
    Args
    ----
    input_hidden_dim : int
        Dimensionality of the input vector from the base model.
    inner_dim : List[int]
        Latent dimensionality of the speculator model.
    vocab_size : int
        Number of entries in the tokenizer associated with the base model.
    n_predict : int
        Number of heads / number of tokens to guess ahead. Model size and speed scale with this value.
    tie_weights : bool
        If true, use a single set of weights for every model head/stage after the first.
        The initial projection from the base model may have a different size, so that stays separate.
    scale_input: bool
        If true, apply an extra layernorm to the initial state vector input.
        Helps training dynamics, particularly when base model output has unusual scale.
    """

    def __init__(self, config: MLPSpeculatorConfig):
        super().__init__()

        self.config = config
        self.n_predict = config.n_predict
        self.input_hidden_dim = config.input_hidden_dim
        inner_dim = [int(i) for i in config.inner_dim.split(".")]
        self.inner_dim = inner_dim
        emb_dim = [int(i) for i in config.emb_dim.split(".")]
        self.emb_dim = emb_dim
        proj_dim = [int(i) for i in config.proj_dim.split(".")]
        self.proj_dim = proj_dim

        self.vocab_size = config.vocab_size
        self.scale_input = config.scale_input
        self.tie_weights = config.tie_weights
        self.tie_lstm_embs = config.tie_lstm_embs
        self.method = config.method
        self.activation = nn.GELU()

        if self.method == "sum_rnn":
            embs = []
            for n_i in range(self.n_predict):
                if not self.tie_weights or n_i == 0:
                    seqs = [nn.Embedding(self.vocab_size, self.emb_dim[0])]
                    for i in range(1, len(self.emb_dim)):
                        print(f"ADDING ANOTHER EMB {i}")
                        seqs.append(
                            LayerNormParameterized(
                                self.emb_dim[i],
                                elementwise_shift=True,
                                elementwise_scale=True,
                            )
                        )
                        seqs.append(self.activation)
                        seqs.append(
                            nn.Linear(self.emb_dim[i - 1], self.emb_dim[i], bias=False)
                        )
                    embs.append(nn.Sequential(*seqs))
            self.emb = nn.ModuleList(embs)

            projs = []
            for n_i in range(self.n_predict):
                if not self.tie_weights or n_i <= 1:
                    seqs = [
                        nn.Linear(
                            (self.input_hidden_dim if n_i == 0 else self.inner_dim[-1]),
                            self.proj_dim[0],
                            bias=False,
                        )
                    ]
                    for i in range(1, len(self.proj_dim)):
                        print(f"ADDING ANOTHER PROJ {i}")
                        seqs.append(
                            LayerNormParameterized(
                                self.proj_dim[i],
                                elementwise_shift=True,
                                elementwise_scale=True,
                            )
                        )
                        seqs.append(self.activation)
                        seqs.append(
                            nn.Linear(
                                self.proj_dim[i - 1], self.proj_dim[i], bias=False
                            )
                        )
                    projs.append(nn.Sequential(*seqs))
            self.proj = nn.ModuleList(projs)

            lns = []
            for n_i in range(self.n_predict):
                if not self.tie_weights or n_i == 0:
                    seqs = [
                        LayerNormParameterized(
                            self.inner_dim[0],
                            elementwise_shift=True,
                            elementwise_scale=True,
                        )
                    ]
                    for i in range(1, len(self.inner_dim)):
                        print(f"ADDING ANOTHER LN {i}")
                        seqs.append(self.activation)
                        seqs.append(
                            nn.Linear(
                                self.inner_dim[i - 1], self.inner_dim[i], bias=False
                            )
                        )
                        seqs.append(
                            LayerNormParameterized(
                                self.inner_dim[i],
                                elementwise_shift=True,
                                elementwise_scale=True,
                            )
                        )
                    lns.append(nn.Sequential(*seqs))
            self.ln = nn.ModuleList(lns)

        elif self.method == "sum_lstm":
            assert self.tie_weights
            self.forget_emb = nn.ModuleList(
                [nn.Embedding(self.vocab_size, self.emb_dim[0])]
            )
            if self.tie_lstm_embs:
                print("TYING LSTM EMBS!!!!!")
                self.input_emb = self.cell_emb = self.output_emb = self.forget_emb
            else:
                self.input_emb = nn.ModuleList(
                    [nn.Embedding(self.vocab_size, self.emb_dim[0])]
                )
                self.cell_emb = nn.ModuleList(
                    [nn.Embedding(self.vocab_size, self.emb_dim[0])]
                )
                self.output_emb = nn.ModuleList(
                    [nn.Embedding(self.vocab_size, self.emb_dim[0])]
                )
            self.forget_proj = nn.ModuleList(
                [
                    nn.Linear(self.input_hidden_dim, self.proj_dim[0], bias=False),
                    nn.Linear(self.inner_dim[-1], self.proj_dim[0], bias=False),
                ]
            )
            self.input_proj = nn.ModuleList(
                [
                    nn.Linear(self.input_hidden_dim, self.proj_dim[0], bias=False),
                    nn.Linear(self.inner_dim[-1], self.proj_dim[0], bias=False),
                ]
            )
            self.cell_proj = nn.ModuleList(
                [
                    nn.Linear(self.input_hidden_dim, self.proj_dim[0], bias=False),
                    nn.Linear(self.inner_dim[-1], self.proj_dim[0], bias=False),
                ]
            )
            self.output_proj = nn.ModuleList(
                [
                    nn.Linear(self.input_hidden_dim, self.proj_dim[0], bias=False),
                    nn.Linear(self.inner_dim[-1], self.proj_dim[0], bias=False),
                ]
            )
            self.cell_ln = nn.ModuleList(
                [
                    LayerNormParameterized(
                        self.inner_dim[0],
                        elementwise_shift=True,
                        elementwise_scale=True,
                    )
                ]
            )
            self.state_ln = nn.ModuleList(
                [
                    LayerNormParameterized(
                        self.inner_dim[0],
                        elementwise_shift=True,
                        elementwise_scale=True,
                    )
                ]
            )

        if self.scale_input:
            self.ln0 = LayerNormParameterized(
                self.input_hidden_dim, elementwise_shift=False, elementwise_scale=False
            )

        self.head = nn.ModuleList(
            [
                nn.Linear(self.inner_dim[-1], self.vocab_size, bias=False)
                for _ in range(self.n_predict)
            ]
        )

        # Weights ensure that state_0 accounts for 50% of state magnitude by final head in expectation
        self.state_weight = 0.5 ** (0.5 / self.n_predict)
        self.emb_weight = math.sqrt((1 - self.state_weight**2) * (self.emb_dim[-1] / 2))

        # Handle weight tying as specified
        if self.tie_weights and self.n_predict > 1:
            # assert (
            #     self.n_predict > 1
            # ), "You cannot tie weights between stages when only 1 exists"

            # for emb in self.emb:
            #     emb.weight = self.emb[0].weight
            #
            for head in self.head:
                head.weight = self.head[0].weight

            # for ln in self.ln:
            #     ln.weight = self.ln[0].weight
            #     ln.bias = self.ln[0].bias

            # Since first proj has different size, allow different initial proj from base into model
            # for i in range(2, self.n_predict):
            #     self.proj[i].weight = self.proj[1].weight

    def reset_parameters(
        self,
        method="zeros",
        model_weight_base_dir="",
    ):
        if model_weight_base_dir:
            weight_map = json.load(
                open(
                    os.path.join(model_weight_base_dir, "model.safetensors.index.json")
                )
            )["weight_map"]
            with safe_open(
                os.path.join(model_weight_base_dir, weight_map["lm_head.weight"]),
                framework="pt",
                device="cpu",
            ) as f:
                lm_head_weight = f.get_tensor("lm_head.weight")
            with safe_open(
                os.path.join(
                    model_weight_base_dir, weight_map["model.embed_tokens.weight"]
                ),
                framework="pt",
                device="cpu",
            ) as f:
                emb_weight = f.get_tensor("model.embed_tokens.weight")

        for n, m in self.named_modules():
            if isinstance(m, LayerNormParameterized) and hasattr(m, "weight"):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if method == "zeros":
                if isinstance(m, nn.Embedding) or isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 1 / math.sqrt(min(m.weight.shape)))
            elif "from_model" in method:
                print(f"INITIALIZING {n} from model")
                if isinstance(m, nn.Embedding):
                    weight_factor = m.weight.shape[-1] // emb_weight.shape[-1]
                    if weight_factor > 1:
                        emb_weight = emb_weight.repeat(1, weight_factor)
                    m.weight.data.copy_(emb_weight)
                if isinstance(m, nn.Linear):
                    if "head" in n:
                        weight_factor = m.weight.shape[-1] // lm_head_weight.shape[-1]
                        if weight_factor > 1:
                            lm_head_weight = (
                                lm_head_weight.repeat(1, weight_factor) / weight_factor
                            )
                        m.weight.data.copy_(lm_head_weight)
                    else:
                        nn.init.normal_(m.weight, 0, 1 / math.sqrt(min(m.weight.shape)))
                        if method == "from_model_else_ones":
                            print(f"INITIALIZING {n} from model adding ones")
                            one_weights = torch.eye(
                                min(m.weight.shape), device=m.weight.device
                            )
                            weight_factor = m.weight.shape[0] // m.weight.shape[1]
                            if weight_factor == 0:
                                raise NotImplementedError
                            if weight_factor > 1:
                                one_weights = one_weights.repeat(weight_factor, 1)
                            m.weight.data.add_(one_weights)

    def generate_suffixes(
        self,
        state: torch.Tensor,
        ind: torch.Tensor,
        topk: list[int] = [5, 4, 3],
        n: int = 5,
    ) -> torch.Tensor:
        """
        FOR INFERENCE
        Generate tree of candidate sequences.
        ...
        Args
        ----
        state : torch.Tensor
            Most recent embedding vector from the base model (pre-classification head).
            Expects size [b 1 d] where b is batch size and d is model width.
        ind : torch.Tensor
            Token indices of the base model's most recent predicted token(s).
            Expects size [b 1] where b is batch size.
        topk : List(int)
            Number of tokens to consider from each head when forming the candidate tree.
            For each candidate branch in the tree, head n produces topk[n] additional sub-branches.
        n : int
            Given the final tree of prod(topk) candidates, return only the top n most confident.
        ...
        Output : torch.Tensor
            The tensor of most likely candidate sequences.
            Has size [b n self.n_predict], where b is batch size and n is provided above.
        """
        # k indicates # of candidates
        # h indicates # of generated tokens
        b = state.size(0)
        k = math.prod(topk)
        out = torch.empty(
            b, 1, k, self.n_predict, device=state.device
        ).int()  # b 1 k h -> b k 1 h
        log_probs = torch.zeros(b, 1, k, device=state.device)  # b 1 k -> b k 1
        assert (
            len(topk) == self.n_predict
        ), f"You must provide a topk number for each head ({self.n_predict} heads, {len(topk)} provided)"
        if self.scale_input:
            state = self.ln0(state) / (2**0.5)
        for i in range(self.n_predict):
            # Project and predict
            z = self.emb[i](ind)  # b k d
            state = self.proj[i](state)
            # Weighted add of state_weight*state and emb_weight*z
            # Let subsequent LN take care of denominator
            # state_weight is close to 1, so shouldn't be any precision issues
            state = torch.add(state, z, alpha=self.emb_weight / self.state_weight)
            state = self.activation(self.ln[i](state))  # b k d
            probs = F.log_softmax(self.head[i](state), dim=2)  # b k v
            probs, preds = probs.topk(topk[i], dim=2)  # b k k'

            # Update candidate set with new predictions, repeating shared prefixes as needed
            out = out.view(b, preds.size(1) * preds.size(2), -1, self.n_predict)
            out[:, :, :, i] = preds.view(b, -1, 1)

            # Update state, log_probs and ind for new predictions
            state = state.unsqueeze(2).expand(-1, -1, topk[i], -1)  # b k k' d
            state = state.reshape(b, -1, state.size(3))  # b kk' d
            ind = preds.view(b, -1)  # b kk'
            log_probs = log_probs.view(b, probs.size(1) * probs.size(2), -1)
            log_probs = log_probs.add(probs.view(b, -1, 1))

        # Take only top n best guesses
        out = out.view(b, k, self.n_predict)
        log_probs = log_probs.view(b, k)
        best_guesses = log_probs.topk(n, dim=1)[1]  # b k
        return out.gather(
            1, best_guesses.unsqueeze(2).expand(-1, -1, self.n_predict)
        )  # b n h

    def forward(
        self,
        state: torch.Tensor,
        inds: torch.Tensor,
    ) -> torch.Tensor:
        """
        FOR TRAINING
        A parallel forward pass on pre-existing ground-truth tokens in pretraining contexts.
        Produces self.n_predict predicted tokens for each token embedding in state.
        Inds requires self.n_predict extra tokens on the right to "simulate" recursive
        behavior for end positions.
        ...
        Args
        ----
        state : torch.Tensor
            Embedding vectors from the base model for a given sequence.
            Expects size [b n d] where b is batch size, n is seq len, and d is model width.
        inds : torch.Tensor
            Ground-truth token indices. inds[:,i] is the prediction coming from state[:,i]
            (or the legal fiction ground truth corresponding to that prediction).
            Expects size [b n+self.n_predict].
        ...
        Output : torch.Tensor
            Prediction logits at each position, for each head of the speculator.
            Has size [self.n_predict b n v] where v is vocab size.
        """
        out = []
        if self.scale_input:
            state = self.ln0(state) / (2**0.5)

        state_shapes = list(state.shape)
        state_shapes[-1] = self.inner_dim[-1]
        if self.method == "sum_lstm":
            cell_state = torch.zeros(
                state_shapes, device=state.device, dtype=state.dtype
            )
            for i in range(self.n_predict):
                prev_state = state
                actual_i = 0 if self.tie_weights else i
                actual_proj_i = 1 if self.tie_weights and i >= 2 else i

                z = self.forget_emb[actual_i](inds[:, i : i + state.size(1)])  # b n d
                state = self.forget_proj[actual_proj_i](prev_state)
                forget_gate = torch.sigmoid(
                    torch.add(state, z, alpha=self.emb_weight / self.state_weight)
                )

                z = self.input_emb[actual_i](inds[:, i : i + state.size(1)])  # b n d
                state = self.input_proj[actual_proj_i](prev_state)
                input_gate = torch.sigmoid(
                    torch.add(state, z, alpha=self.emb_weight / self.state_weight)
                )

                z = self.cell_emb[actual_i](inds[:, i : i + state.size(1)])  # b n d
                state = self.cell_proj[actual_proj_i](prev_state)
                cell_candidate = torch.add(
                    state, z, alpha=self.emb_weight / self.state_weight
                )
                cell_candidate = self.activation(
                    self.cell_ln[actual_i](cell_candidate)
                )  # b n d
                cell_candidate = cell_candidate * input_gate

                z = self.output_emb[actual_i](inds[:, i : i + state.size(1)])  # b n d
                state = self.output_proj[actual_proj_i](prev_state)
                output_gate = torch.sigmoid(
                    torch.add(state, z, alpha=self.emb_weight / self.state_weight)
                )

                cell_state = cell_state * forget_gate
                cell_state = cell_state + cell_candidate

                state_candidate = self.activation(self.state_ln[actual_i](cell_state))
                state = state_candidate * output_gate

                # Weighted add of state_weight*state and emb_weight*z
                # Let subsequent LN take care of denominator
                # state_weight is close to 1, so shouldn't be any precision issues
                out.append(self.head[i](state))  # b n v

        else:
            assert self.method == "sum_rnn"
            for i in range(self.n_predict):
                actual_i = 0 if self.tie_weights else i
                actual_proj_i = 1 if self.tie_weights and i >= 2 else i

                z = self.emb[actual_i](inds[:, i : i + state.size(1)])  # b n d
                state = self.proj[actual_proj_i](state)
                # Weighted add of state_weight*state and emb_weight*z
                # Let subsequent LN take care of denominator
                # state_weight is close to 1, so shouldn't be any precision issues
                state = torch.add(state, z, alpha=self.emb_weight / self.state_weight)
                state = self.activation(self.ln[actual_i](state))  # b n d
                out.append(self.head[i](state))  # b n v

        return torch.stack(out, dim=0)  # h b n v
