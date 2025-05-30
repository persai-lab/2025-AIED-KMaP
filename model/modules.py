import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from torch import nn


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.net(x)
        out = self.norm(out + x)

        return out


class Attention(nn.Module):

    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()

        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.reshape_head_q = nn.Linear(dim, inner_dim)
        self.reshape_head_k = nn.Linear(dim, inner_dim)
        self.reshape_head_v = nn.Linear(dim, inner_dim)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, q, k, v):
        q = self.reshape_head_q(q)
        k = self.reshape_head_k(k)
        v = self.reshape_head_v(v)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MultiHeadAttentionModule(nn.Module):

    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()

        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, dim_head, dropout)

        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, context):
        x = self.attn(q, context, context) * context
        x = self.norm(x)
        x = self.ff(x) + x
        x = self.norm2(x)

        return x


class SoftKmeans:

    def __init__(self, num_clusters, device):
        super().__init__()

        self.num_clusters = num_clusters
        self.device = device

    def normalize(self, embeddings, student_ids, unique_students):
        for student in unique_students:
            mask = student_ids == student
            embeddings[mask] = F.normalize(embeddings[mask], p=2, dim=-1)

        return embeddings

    def aggregate_embeddings(self, embeddings, student_ids, unique_students):
        aggregated_embeddings = []
        for student in unique_students:
            mask = student_ids == student
            student_embeddings = embeddings[mask]
            aggregated_embedding = student_embeddings.mean(dim=0)

            aggregated_embeddings.append(aggregated_embedding)

        return torch.stack(aggregated_embeddings)

    def apply_constraints(self, embeddings, student_ids, behavior_embeddings, temperature=0.1, eps=1e-8):
        unique_students = torch.unique(student_ids)
        
        embeddings = self.normalize(embeddings, student_ids, unique_students)
        behavior_embeddings = self.normalize(behavior_embeddings, student_ids, unique_students)

        aggregated_embeddings = self.aggregate_embeddings(embeddings, student_ids, unique_students)
        aggregated_behavior_embeddings = self.aggregate_embeddings(behavior_embeddings, student_ids, unique_students)

        local_kmeans = KMeans(self.num_clusters, random_state=42)
        cluster_assig = local_kmeans.fit_predict(aggregated_behavior_embeddings.detach().cpu())
        clf = NearestCentroid()
        clf.fit(aggregated_embeddings.detach().cpu().numpy(), cluster_assig)
        cluster_centroids = torch.tensor(clf.centroids_, device=self.device).to(torch.float32)
        cluster_distances = torch.cdist(aggregated_embeddings, cluster_centroids, p=2) 
        cluster_assignments = cluster_distances.argmin(1)
        
        cluster_ids = torch.arange(0, self.num_clusters)
        intra_cluster_distances = torch.zeros(aggregated_embeddings.shape[0], device=self.device)
        nearest_cluster_distances = torch.zeros(aggregated_embeddings.shape[0], device=self.device)
        distances = torch.cdist(aggregated_embeddings, aggregated_embeddings)
        for cluster_id in range(self.num_clusters):
            mask = (cluster_assignments == cluster_id)
            if mask.sum() > 0:
                intra_cluster_distances[mask] = distances[mask][:, mask].mean(dim=1)
                nearest_cluster_distances[mask] = cluster_distances[mask][:, cluster_ids != cluster_id].min(dim=1)[0]

        sil_loss = (nearest_cluster_distances - intra_cluster_distances) / (torch.maximum(intra_cluster_distances, nearest_cluster_distances) + eps)
        sil_loss = torch.exp(-sil_loss / temperature)

        distances = torch.cdist(embeddings, embeddings, p=2)
        distances = torch.triu(distances, diagonal=1)
        user_loss = distances[student_ids.unsqueeze(1) == student_ids.unsqueeze(0)].sum() / (len(student_ids) * (len(student_ids)-1) / 2)

        total_loss = (sil_loss.mean() + user_loss) / 2

        print("Total clustering loss", total_loss.detach().item())

        return total_loss, (unique_students, cluster_assignments)
