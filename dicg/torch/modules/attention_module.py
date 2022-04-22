import torch
import torch.nn as nn
import torch.nn.functional as F


from dicg.utils import get_device


class AttentionModule(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`
    """

    def __init__(self, dimensions, attention_type="general", n_agent=None):
        super().__init__()

        self.attention_type = attention_type
        if self.attention_type == "general":
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)
        elif self.attention_type == "diff":
            self.linear_in = nn.Linear(dimensions, 1, bias=False)
        elif self.attention_type == "hard_soft":
            self.comm_rnn = nn.GRU(2 * dimensions, dimensions, bidirectional=True)
            self.hard_encoding = nn.Linear(2 * dimensions, 2, bias=False)
            self.soft_encoding = nn.Linear(dimensions, dimensions, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query):
        """
            Self attention

            n_paths, max_path_length, n_agents, emb_feat_dim = query.shape
            OR
            bs, n_agents, emb_feat_dim = query.shape
            OR
            n_agents, emb_feat_dim = query.shape

        """

        if self.attention_type in ["general", "dot"]:
            context = query.transpose(-2, -1).contiguous()
            if self.attention_type == "general":
                query = self.linear_in(query)
            attention_scores = torch.matmul(query, context)
            attention_weights = self.softmax(attention_scores)

        elif self.attention_type == "diff":
            """
                Symmetric
                Kind of unstable
            """
            n_agents = query.shape[-2]
            repeats = (1,) * (len(query.shape) - 2) + (n_agents, 1)
            augmented_shape = query.shape[:-1] + (n_agents,) + query.shape[-1:]
            # Change query shape to (..., n_agents, n_agents, emb_feat_dim)
            query = query.repeat(*repeats).reshape(*augmented_shape)
            context = query.transpose(-3, -2).contiguous()

            attention_scores = torch.abs(query - context)
            attention_scores = self.linear_in(attention_scores).squeeze(-1)
            attention_scores = torch.tanh(attention_scores)
            attention_weights = self.softmax(attention_scores)

        elif self.attention_type == "identity":
            n_agents = query.shape[-2]
            attention_weights = torch.zeros(
                query.shape[:-2] + (n_agents, n_agents), device=get_device()
            )
            attention_weights.reshape(-1, n_agents, n_agents)
            for i in range(n_agents):
                if len(query.shape) > 2:
                    attention_weights[:, i, i] = 1
                else:
                    attention_weights[i, i] = 1
            attention_weights = attention_weights.reshape(
                query.shape[:-2] + (n_agents, n_agents)
            )

        elif self.attention_type == "uniform":
            n_agents = query.shape[-2]
            attention_weights = torch.ones(query.shape[:-2] + (n_agents, n_agents))
            attention_weights = attention_weights / n_agents

        elif self.attention_type == "hard_soft":
            n_paths, max_path_length, n_agents, emb_feat_dim = query.shape

            # soft
            context = query.transpose(-2, -1).contiguous()
            query = self.linear_in(query)
            attention_scores = torch.matmul(query, context)
            soft_weights = self.softmax(attention_scores)

            # hard
            in_comm = torch.zeros(
                (n_agents - 1, n_paths * max_path_length, n_agents, 2 * emb_feat_dim),
                device=get_device(),
            )
            h = query.reshape(-2, n_agents, emb_feat_dim)
            for i in range(n_agents):
                h_i = h[:, i, :].reshape(1, -1, emb_feat_dim)
                h_i = h_i.repeat(n_agents - 1, 1, 1)
                h_j = h[:, [j for j in range(n_agents) if j != i], :].permute(1, 0, 2)
                in_comm[:, :, i, :] = torch.cat((h_i, h_j), dim=-1)
            in_comm = in_comm.reshape(n_agents - 1, -1, 2 * emb_feat_dim)
            out_comm = torch.zeros(
                (2 * 1, n_paths * max_path_length * n_agents, emb_feat_dim),
                device=get_device(),
            )
            # (n_agents-1, n_paths*max_path_length*n_agents, 2*emb_feat_dim)
            out_comm, _ = self.comm_rnn(in_comm, out_comm)
            # (n_paths*max_path_length*n_agents*n_agents-1, 2*emb_feat_dim)
            out_comm = out_comm.permute(1, 0, 2).reshape(-1, 2 * emb_feat_dim)

            hard_weights = torch.ones_like(soft_weights)

            hard_weights_temp = self.hard_encoding(out_comm)
            hard_weights_temp = F.gumbel_softmax(hard_weights_temp, tau=0.01)
            hard_weights_temp = hard_weights_temp[:, 1].reshape(
                n_paths, max_path_length, n_agents, n_agents - 1
            )
            for i in range(n_agents):
                hard_weights[
                    :, :, i, [j for j in range(n_agents) if j != i]
                ] = hard_weights_temp[:, :, i, :]

            attention_weights = hard_weights * soft_weights

        return attention_weights

