import torch
import torch.nn as nn


class Codebook(nn.Module):
    """
    Codebook mapping: takes in an encoded image and maps each vector onto its closest codebook vector.
    Metric: mean squared error = (z_e - z_q)**2 = (z_e**2) - (2*z_e*z_q) + (z_q**2)
    """

    def __init__(self, args):
        super().__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        q_loss_encoder = torch.mean((z_q.detach() - z) ** 2)
        q_loss_codebook = torch.mean((z_q - z.detach()) ** 2)
        loss = self.beta*q_loss_encoder +  q_loss_codebook

        # preserve gradients
        z_q = z + (z_q - z).detach()  # moving average instead of hard codebook remapping

        z_q = z_q.permute(0, 3, 1, 2)

        q_loss_dict = {
            "q_loss": loss.item(),
            "q_loss_encoder": q_loss_encoder.item(),
            "q_loss_codebook": q_loss_codebook.item(),
        }
        return z_q, min_encoding_indices, q_loss_dict