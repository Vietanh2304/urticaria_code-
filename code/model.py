# model.py
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

# Import cfg từ config.py để các lớp model có thể truy cập
from config import cfg

# ===============================================================
# ĐỊNH NGHĨA MODEL (G_MMNet)
# ===============================================================
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, img_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        return x

class CrossScaleBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, n_heads, dropout):
        super().__init__()
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.q_proj = nn.Linear(dim_q, dim_kv)
        self.kv_proj = nn.Linear(dim_kv, dim_kv)
        self.out_proj = nn.Linear(dim_kv, dim_q)
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.attn = nn.MultiheadAttention(dim_kv, n_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim_q)
        self.ffn = nn.Sequential(
            nn.Linear(dim_q, dim_q * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_q * 4, dim_q), nn.Dropout(dropout)
        )

    def forward(self, query_stream, kv_stream):
        q = self.q_proj(self.norm_q(query_stream))
        kv = self.kv_proj(self.norm_kv(kv_stream))
        attn_out, _ = self.attn(q, kv, kv)
        attn_out = self.out_proj(attn_out)
        query_stream = query_stream + attn_out
        query_stream = query_stream + self.ffn(self.norm_ffn(query_stream))
        return query_stream

class FocusedDualScaleImageEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim_s1, dim_s2 = cfg.IMG_ENCODER_EMBED_DIM_S1, cfg.IMG_ENCODER_EMBED_DIM_S2
        self.patch_embed_s1 = PatchEmbedding(3, 8, dim_s1, cfg.IMG_SIZE)
        self.patch_embed_s2 = PatchEmbedding(3, 16, dim_s2, cfg.IMG_SIZE)
        self.cross_attn_s1_queries_s2 = CrossScaleBlock(dim_q=dim_s1, dim_kv=dim_s2, n_heads=4, dropout=cfg.DROPOUT)
        self.cross_attn_s2_queries_s1 = CrossScaleBlock(dim_q=dim_s2, dim_kv=dim_s1, n_heads=4, dropout=cfg.DROPOUT)
        self.fusion_norm = nn.LayerNorm(dim_s1 + dim_s2)
        self.final_proj = nn.Linear(dim_s1 + dim_s2, cfg.IMG_ENCODER_FINAL_DIM)

    def forward(self, img):
        tokens_s1 = self.patch_embed_s1(img)
        tokens_s2 = self.patch_embed_s2(img)
        refined_s1 = self.cross_attn_s1_queries_s2(tokens_s1, tokens_s2)
        refined_s2 = self.cross_attn_s2_queries_s1(tokens_s2, tokens_s1)
        cls_s1 = refined_s1[:, 0, :]
        cls_s2 = refined_s2[:, 0, :]
        combined_cls = torch.cat([cls_s1, cls_s2], dim=1)
        return self.final_proj(self.fusion_norm(combined_cls))

class FeatureRelationModule(nn.Module):
    def __init__(self, num_features, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_features * dim, num_features * dim), nn.GELU(),
            nn.Linear(num_features * dim, num_features * num_features)
        )
        self.scale_factor = nn.Parameter(torch.tensor(0.01))
    def forward(self, node_features):
        b, n, d = node_features.shape
        R = self.mlp(node_features.view(b, -1)).view(b, n, n)
        return R * self.scale_factor

class CSACM_Encoder(nn.Module):
    def __init__(self, *, cat_dims, num_continuous, dim, depth, heads):
        super().__init__()
        self.num_continuous = num_continuous
        self.cat_embeds = nn.ModuleList([nn.Embedding(cardinality, dim) for cardinality in cat_dims])
        self.cont_embed = nn.Linear(1, dim)
        num_features = num_continuous + len(cat_dims)
        self.frm = FeatureRelationModule(num_features, dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True, dropout=cfg.DROPOUT)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.graph_cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
    def forward(self, x_meta):
        x_cont, x_cat = x_meta[:, :self.num_continuous], x_meta[:, self.num_continuous:].long()
        cat_embs = [embed(x_cat[:, i]).unsqueeze(1) for i, embed in enumerate(self.cat_embeds)]
        cont_embs = [self.cont_embed(x_cont[:, i].unsqueeze(-1)).unsqueeze(1) for i in range(x_cont.shape[1])]
        node_features = torch.cat(cont_embs + cat_embs, dim=1)
        R_matrix = self.frm(node_features)
        weighted_features = node_features + torch.bmm(R_matrix.softmax(dim=2), node_features)
        node_features = self.transformer_encoder(weighted_features)
        cls_token = self.graph_cls_token.repeat(node_features.shape[0], 1, 1)
        meta_summary, _ = self.cls_attn(cls_token, node_features, node_features)
        return meta_summary

class CoAttentionBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.img_attn, self.meta_attn = [nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(2)]
        self.ffn_img = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.ffn_meta = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.norm1, self.norm2, self.norm3, self.norm4 = [nn.LayerNorm(dim) for _ in range(4)]
    def forward(self, img_tokens, meta_tokens):
        img_refined = self.img_attn(self.norm1(img_tokens), self.norm2(meta_tokens), self.norm2(meta_tokens))[0]
        img_tokens = img_tokens + img_refined
        img_tokens = img_tokens + self.ffn_img(self.norm3(img_tokens))
        meta_refined = self.meta_attn(self.norm2(meta_tokens), self.norm1(img_tokens), self.norm1(img_tokens))[0]
        meta_tokens = meta_tokens + meta_refined
        meta_tokens = meta_tokens + self.ffn_meta(self.norm4(meta_tokens))
        return img_tokens, meta_tokens

class G_MMNet(nn.Module):
    def __init__(self, cfg, cat_dims, num_continuous):
        super().__init__()
        self.img_encoder = FocusedDualScaleImageEncoder(cfg)
        meta_dim = cfg.IMG_ENCODER_FINAL_DIM
        self.meta_encoder = CSACM_Encoder(cat_dims=cat_dims, num_continuous=num_continuous,
                                          dim=meta_dim, depth=cfg.META_DEPTH, heads=cfg.META_HEADS)
        self.co_attention_blocks = nn.ModuleList(
            [CoAttentionBlock(meta_dim, cfg.CO_ATTENTION_HEADS) for _ in range(cfg.CO_ATTENTION_DEPTH)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(meta_dim * 2), 
            nn.Linear(meta_dim * 2, meta_dim),
            nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(meta_dim, cfg.NUM_CLASSES)
        )
        
    def forward(self, img, meta):
        img_rep = self.img_encoder(img)
        meta_summary_token = self.meta_encoder(meta)
        img_token = img_rep.unsqueeze(1)
        for block in self.co_attention_blocks:
            img_token, meta_summary_token = block(img_token, meta_summary_token)
        img_rep_final = img_token.squeeze(1)
        meta_rep_final = meta_summary_token.squeeze(1)
        combined = torch.cat([img_rep_final, meta_rep_final], dim=1)
        return self.classifier(combined)