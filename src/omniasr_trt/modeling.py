"""omniASR models (wav2vec2 encoder, CTC head, LLaMA decoder) in plain PyTorch.

Module names follow the fairseq2 checkpoints (``state_dict["model"]``), so the official files load with
``strict=True``; the only conversion is folding the positional convolution's weight norm.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class EncoderConfig:
    model_dim: int = 2048
    num_layers: int = 128
    num_heads: int = 16
    ffn_dim: int = 8192
    feature_dim: int = 512
    # (channels, kernel, stride) of each feature-extractor convolution
    conv_layers: tuple = ((512, 10, 5),) + ((512, 3, 2),) * 4 + ((512, 2, 2),) * 2
    pos_kernel: int = 128
    pos_groups: int = 16


@dataclass(frozen=True)
class DecoderConfig:
    model_dim: int = 4096
    num_layers: int = 12
    num_heads: int = 8
    ffn_dim: int = 2816
    vocab_size: int = 10288
    num_special: int = 3  # lid marker, last segment, regular segment (text embedding rows beyond the vocab)
    num_langs: int = 1694
    max_seq_len: int = 8192
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-5


HOP = 320  # samples per encoder frame at 16 kHz
# The feature extractor's activations (512 channels at 3.2 kHz, with fp32 copies for its LayerNorms) are the encoder's
# memory peak; it runs on at most this many samples at a time, the transformer on the whole batch.
FEATURE_CHUNK_SAMPLES = 120 * 16000
# Padded encoder batches: "runs" attends per run of equal-length rows without a mask (flash kernels), "mask" with a
# key mask (slower kernels, one call), "auto" runs when there are at most RUNS_MAX runs. With flash attention, runs
# are the fastest even when every row has its own length (319 vs 260 audio s/s on 206 inputs <= 90 s, H200).
PADDED_ATTENTION = "runs"
RUNS_MAX = 3
# SDPA backends for the encoder, first applicable wins (None: PyTorch's choice). PyTorch's default, cuDNN, builds an
# execution plan for every new sequence length, which arbitrary real-world lengths pay again and again (204 audio
# s/s on a first pass vs 342 on a second); flash does not (319 / 325). fp32 falls through to the others.
# "auto": PyTorch's choice for unpadded batch shapes that recur (seen RECURRING times: 15 s LLM segments, 30 s CTC
# windows; their cached cuDNN plan is the fastest, +4% on long-form CTC), flash for the others. Padded batches
# practically never repeat their lengths on real data; a plan for a shape seen a few times costs more than it saves
# (229 instead of 319 audio s/s once a benchmark replays the same padded batches).
ENCODER_SDPA: list | str | None = "auto"
FLASH_FIRST = ["flash", "efficient", "math"]
RECURRING = 3
_shape_counts: dict = {}


def conv_lengths(lengths: Tensor, layers) -> Tensor:
    """Valid frames after each feature-extractor convolution (fairseq2: floor((L - k) / s + 1))."""
    for _, kernel, stride in layers:
        lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
    return lengths


class Fp32LayerNorm(nn.LayerNorm):
    """LayerNorm computed in fp32 and cast back (fairseq2 ``cast_fp32=True``)."""

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x.float(), self.normalized_shape, self.weight.float(), self.bias.float(),
                            self.eps).type_as(x)


class FeatureExtractionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel, stride=stride, bias=True)
        self.layer_norm = Fp32LayerNorm(out_dim)

    def forward(self, x: Tensor) -> Tensor:  # [B, C, T]
        x = self.conv(x)
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        return F.gelu(x)


class FeatureExtractor(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        dims = [1] + [c for c, _, _ in cfg.conv_layers]
        self.layers = nn.ModuleList(
            FeatureExtractionLayer(dims[i], c, k, s) for i, (c, k, s) in enumerate(cfg.conv_layers))

    def forward(self, waves: Tensor) -> Tensor:  # [B, S] -> [B, T, C]
        x = waves.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)


class PositionEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.conv = nn.Conv1d(cfg.model_dim, cfg.model_dim, cfg.pos_kernel, padding=cfg.pos_kernel // 2,
                              groups=cfg.pos_groups, bias=True)
        self.remove_pad = cfg.pos_kernel % 2 == 0

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:  # [B, T, D], mask [B, T] True = valid
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        enc = self.conv(x.transpose(1, 2))
        if self.remove_pad:
            enc = enc[:, :, :-1]
        return x + F.gelu(enc).transpose(1, 2)


class EncoderFrontend(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.feature_extractor = FeatureExtractor(cfg)
        self.post_extract_layer_norm = nn.LayerNorm(cfg.feature_dim)
        self.model_dim_proj = nn.Linear(cfg.feature_dim, cfg.model_dim)
        self.pos_encoder = PositionEncoder(cfg)


class EncoderAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.qkv: tuple[Tensor, Tensor] | None = None  # fused weight and bias (fuse_encoder)

    def forward(self, x: Tensor, padding) -> Tensor:
        """``padding``: None (no padding), a list of ``(first_row, end_row, frames)`` runs of equal length (each
        run attends without a mask, the flash kernels' fast path), or a boolean key mask [B, 1, 1, T]."""
        b, t, d = x.shape
        if self.qkv is not None:
            q, k, v = F.linear(x, *self.qkv).view(b, t, 3, self.heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        else:
            q, k, v = (p(x).view(b, t, self.heads, -1).transpose(1, 2)
                       for p in (self.q_proj, self.k_proj, self.v_proj))
        # The SDPA backends are chosen per batch by Wav2Vec2.encode (ENCODER_SDPA).
        if padding is None or isinstance(padding, Tensor):
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=padding)
        else:
            o = torch.zeros_like(q)  # padded query positions: zeros (never read by a valid position)
            for lo, hi, n in padding:
                o[lo:hi, :, :n] = F.scaled_dot_product_attention(q[lo:hi, :, :n], k[lo:hi, :, :n], v[lo:hi, :, :n])
        return self.output_proj(o.transpose(1, 2).reshape(b, t, d))


def sdpa_backends(names: list | None):
    """``torch.nn.attention.sdpa_kernel`` for backend names ("flash", "efficient", "cudnn", "math"), or a no-op."""
    if not names:
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    table = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
             "cudnn": SDPBackend.CUDNN_ATTENTION, "math": SDPBackend.MATH}
    return sdpa_kernel([table[n] for n in names])


class EncoderFFN(nn.Module):
    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.inner_proj = nn.Linear(dim, inner)
        self.output_proj = nn.Linear(inner, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_proj(F.gelu(self.inner_proj(x)))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(cfg.model_dim)
        self.self_attn = EncoderAttention(cfg.model_dim, cfg.num_heads)
        self.ffn_layer_norm = nn.LayerNorm(cfg.model_dim)
        self.ffn = EncoderFFN(cfg.model_dim, cfg.ffn_dim)

    def forward(self, x: Tensor, attn_mask: Tensor | None) -> Tensor:
        x = x + self.self_attn(self.self_attn_layer_norm(x), attn_mask)
        return x + self.ffn(self.ffn_layer_norm(x))


class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.num_layers))
        self.layer_norm = nn.LayerNorm(cfg.model_dim)


class Wav2Vec2(nn.Module):
    """The shared wav2vec2 encoder: normalised 16 kHz waves -> frame features ``[B, T, 2048]``."""

    def __init__(self, cfg: EncoderConfig = EncoderConfig()):
        super().__init__()
        self.enc_cfg = cfg
        self.encoder_frontend = EncoderFrontend(cfg)
        self.encoder = Encoder(cfg)

    def frame_lengths(self, lengths: Tensor) -> Tensor:
        return conv_lengths(lengths, self.enc_cfg.conv_layers)

    def encode(self, waves: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """``waves`` [B, S] (already normalised, in the model dtype), ``lengths`` [B] on the CPU; returns features
        and frame lengths (CPU). Nothing here waits for the GPU, so it can be queued on a side stream."""
        fe = self.encoder_frontend
        rows = max(1, FEATURE_CHUNK_SAMPLES // max(1, waves.size(1)))
        x = torch.cat([fe.feature_extractor(waves[i : i + rows]) for i in range(0, waves.size(0), rows)])
        t = x.size(1)
        if lengths is None:
            frames = torch.full((x.size(0),), t, dtype=torch.long)
        else:
            frames = self.frame_lengths(lengths.cpu()).clamp(max=t)
        mask = padding = None
        if bool((frames < t).any()):
            mask = torch.arange(t, device=x.device)[None, :] < frames.to(x.device)[:, None]  # [B, T]
            r = runs(frames.tolist()) if PADDED_ATTENTION != "mask" else None
            if r is not None and PADDED_ATTENTION == "auto" and len(r) > RUNS_MAX:
                r = None
            padding = r or mask[:, None, None, :]
        x = fe.model_dim_proj(fe.post_extract_layer_norm(x))
        x = fe.pos_encoder(x, mask)
        backends = ENCODER_SDPA
        if backends == "auto":
            backends = FLASH_FIRST
            if padding is None:
                shape = (x.size(0), t)
                _shape_counts[shape] = _shape_counts.get(shape, 0) + 1
                if _shape_counts[shape] >= RECURRING:
                    backends = None
        with sdpa_backends(backends):
            for layer in self.encoder.layers:
                x = layer(x, padding)
        return self.encoder.layer_norm(x), frames


def runs(lengths: list[int]) -> list[tuple[int, int, int]] | None:
    """Rows of equal length as contiguous ``(lo, hi, length)`` runs, or None when equal lengths are not adjacent
    (callers batch longest first, so they are)."""
    out = []
    for i, n in enumerate(lengths):
        if out and out[-1][2] == n:
            out[-1] = (out[-1][0], i + 1, n)
        elif any(r[2] == n for r in out):
            return None
        else:
            out.append((i, i + 1, n))
    return out


def fuse_encoder(model: Wav2Vec2) -> None:
    """One q|k|v projection per layer; the separate modules' weights become slices of it (no second copy)."""
    for layer in model.encoder.layers:
        a = layer.self_attn
        if a.qkv is not None:
            continue
        with torch.no_grad():
            w = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
            bias = torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias])
        d = a.q_proj.weight.size(0)
        for j, proj in enumerate((a.q_proj, a.k_proj, a.v_proj)):
            proj.weight = nn.Parameter(w[j * d : (j + 1) * d], requires_grad=False)
            proj.bias = nn.Parameter(bias[j * d : (j + 1) * d], requires_grad=False)
        a.qkv = (w, bias)


class CTCModel(Wav2Vec2):
    """omniASR CTC: encoder + ``final_proj`` to the 10288 characters."""

    def __init__(self, cfg: EncoderConfig = EncoderConfig(), vocab_size: int = 10288):
        super().__init__(cfg)
        self.final_proj = nn.Linear(cfg.model_dim, vocab_size)

    def forward(self, waves: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x, frames = self.encode(waves, lengths)
        return self.final_proj(x), frames


# ---------------------------------------------------------------------------------------------- LLaMA decoder


def rope_table(cfg: DecoderConfig, device=None) -> Tensor:
    """cos/sin of every position, ``[max_seq_len, head_dim / 2, 2]`` fp32 (fairseq2's RotaryEncoder)."""
    head_dim = cfg.model_dim // cfg.num_heads
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(cfg.max_seq_len, device=device, dtype=torch.float32), inv)
    return torch.stack([torch.cos(angles), torch.sin(angles)], -1)


def apply_rope(x: Tensor, cs: Tensor) -> Tensor:
    """Interleaved-pair rotation in fp32. ``x`` [..., S, H, D], ``cs`` broadcastable to [..., S, 1, D/2, 2]."""
    xf = x.float().unflatten(-1, (-1, 2))
    re, im = xf[..., 0], xf[..., 1]
    c, s = cs[..., 0], cs[..., 1]
    return torch.stack([re * c - im * s, re * s + im * c], -1).flatten(-2).type_as(x)


class DecoderAttention(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        d = cfg.model_dim
        self.heads = cfg.num_heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.output_proj = nn.Linear(d, d, bias=False)


class GLUFFN(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.model_dim, cfg.ffn_dim, bias=False)
        self.inner_proj = nn.Linear(cfg.model_dim, cfg.ffn_dim, bias=False)
        self.output_proj = nn.Linear(cfg.ffn_dim, cfg.model_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_proj(self.inner_proj(x) * F.silu(self.gate_proj(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.self_attn_layer_norm = RMSNorm(cfg.model_dim, cfg.rms_eps)
        self.self_attn = DecoderAttention(cfg)
        self.ffn_layer_norm = RMSNorm(cfg.model_dim, cfg.rms_eps)
        self.ffn = GLUFFN(cfg)


class LlamaDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_layers))
        self.layer_norm = RMSNorm(cfg.model_dim, cfg.rms_eps)


class LLMModel(Wav2Vec2):
    """omniASR LLM: encoder, ``encoder_proj`` to the decoder width, LLaMA decoder over [lang, audio, text]."""

    def __init__(self, cfg: EncoderConfig = EncoderConfig(), dcfg: DecoderConfig = DecoderConfig()):
        super().__init__(cfg)
        self.dec_cfg = dcfg
        self.encoder_proj = nn.Linear(cfg.model_dim, dcfg.model_dim)
        self.text_frontend = nn.Embedding(dcfg.vocab_size + dcfg.num_special, dcfg.model_dim)
        self.llama_decoder = LlamaDecoder(dcfg)
        self.final_proj = nn.Linear(dcfg.model_dim, dcfg.vocab_size, bias=False)
        self.lang_embeddings = nn.Embedding(dcfg.num_langs, dcfg.model_dim)
        self.register_buffer("rope", rope_table(dcfg), persistent=False)

    def embed_audio(self, waves: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x, frames = self.encode(waves, lengths)
        return self.encoder_proj(x), frames

    def decoder_forward(self, x: Tensor, positions: Tensor, cache: list | None = None,
                        cache_lens: Tensor | None = None, causal: bool = True) -> Tensor:
        """Reference-semantics decoder pass (no padding inside a row).

        ``x`` [B, S, D] embeddings at ``positions`` [S] (shared by the batch). With ``cache`` (a list of
        per-layer ``[k, v]`` of shape [B, H, S_past, Dh]) the new keys are appended and attention covers
        the past plus the new tokens (causal within the new ones).
        """
        h = x
        cs = self.rope[positions][:, None]  # [S, 1, D/2, 2]
        b, s, _ = x.shape
        for i, layer in enumerate(self.llama_decoder.layers):
            a = layer.self_attn
            n = layer.self_attn_layer_norm(h)
            q = apply_rope(a.q_proj(n).view(b, s, a.heads, -1), cs).transpose(1, 2)
            k = apply_rope(a.k_proj(n).view(b, s, a.heads, -1), cs).transpose(1, 2)
            v = a.v_proj(n).view(b, s, a.heads, -1).transpose(1, 2)
            if cache is not None:
                if cache[i] is not None:
                    k = torch.cat([cache[i][0], k], 2)
                    v = torch.cat([cache[i][1], v], 2)
                cache[i] = [k, v]
            if s == 1:
                o = F.scaled_dot_product_attention(q, k, v)
            elif k.size(2) == s:
                o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            else:  # new block after a past: causal aligned to the end
                past = k.size(2) - s
                mask = torch.ones(s, k.size(2), dtype=torch.bool, device=x.device).tril(past)
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            h = h + a.output_proj(o.transpose(1, 2).reshape(b, s, -1))
            h = h + layer.ffn(layer.ffn_layer_norm(h))
        return self.llama_decoder.layer_norm(h)


# ---------------------------------------------------------------------------------------------- weights


def fold_weight_norm(g: Tensor, v: Tensor, dim: int = 2) -> Tensor:
    """``torch.nn.utils.weight_norm`` weight: ``g * v / ||v||`` with the norm over every dim but ``dim``."""
    dims = [d for d in range(v.dim()) if d != dim]
    return g * (v / v.norm(dim=dims, keepdim=True))


def convert_state_dict(state: dict) -> dict:
    """fairseq2 checkpoint (``{"model": ..., "fs2": ...}`` or the bare dict) -> this module's state dict."""
    state = dict(state.get("model", state))
    prefix = "encoder_frontend.pos_encoder.conv."
    g, v = state.pop(prefix + "weight_g", None), state.pop(prefix + "weight_v", None)
    if g is not None:
        state[prefix + "weight"] = fold_weight_norm(g.float(), v.float()).to(v.dtype)
    return state


def load_checkpoint(model: nn.Module, path, dtype: torch.dtype, device="cuda") -> nn.Module:
    """Official ``.pt`` (fp32, fairseq2 names) or a converted ``.safetensors`` -> ``model`` (strict)."""
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path, device="cpu")
    else:
        state = convert_state_dict(torch.load(path, map_location="cpu", mmap=True, weights_only=True))
    model.load_state_dict(state, strict=True, assign=True)
    model = model.to(device=device, dtype=dtype).eval()
    if isinstance(model, LLMModel):  # .to(dtype) cast the fp32 rotation table
        model.rope = rope_table(model.dec_cfg, device)
    fuse_encoder(model)
    return model


def convert(src, dst, dtype: torch.dtype = torch.bfloat16) -> None:
    """Official ``.pt`` -> ``.safetensors`` in ``dtype`` with the weight norm folded (faster to load)."""
    from safetensors.torch import save_file

    state = convert_state_dict(torch.load(str(src), map_location="cpu", mmap=True, weights_only=True))
    save_file({k: v.to(dtype).contiguous() for k, v in state.items()}, str(dst))


def build(kind: str, meta: bool = True) -> nn.Module:
    with torch.device("meta") if meta else torch.device("cpu"):
        model = CTCModel() if kind == "ctc" else LLMModel()
    if meta and kind == "llm":  # non-persistent buffer: recompute on the CPU
        model.rope = rope_table(model.dec_cfg)
    return model


def normalize(wave: Tensor) -> Tensor:
    """Zero mean, unit variance over the whole input in fp32 (omnilingual-asr's ``layer_norm(wave, wave.shape)``)."""
    wave = wave.float()
    return F.layer_norm(wave, wave.shape)
