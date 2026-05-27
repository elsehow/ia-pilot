"""Gauge-transform math for semantics-preserving basis changes.

Three independent gauges available:

1. Q/K gauge (build_gauge_block + apply_gauge_to_layer).
   Acts on the (q,k) basis. Must commute with RoPE, so it's constrained to be
   block-diagonal in the NeoX RoPE layout (2D rotation per pair). Also: only
   works on models WITHOUT qk-norm (Llama yes, Qwen3 no).

2. V/O gauge (build_orthogonal_gauge + apply_vo_gauge_to_layer).
   Acts on the value-vector basis. No RoPE constraint and no qk-norm
   interaction, so G can be any invertible matrix per head. Works on both
   Llama AND Qwen3.

3. MLP hidden-dim permutation (build_random_permutation +
   apply_mlp_permutation_to_layer). For SiLU-gated MLPs (Llama, Qwen3,
   Mistral, ...), permuting the columns of W_up and W_gate by the same π
   and the rows of W_down by π is exactly output-preserving. Targets the
   IA's MLP-side LoRA (up_proj / gate_proj / down_proj) — the surfaces
   that V/O gauge leaves untouched. Universal: no RoPE/qk-norm constraint.

All three are output-preserving in fp32; bf16 round-trip introduces a small
floating-point error. The attack composes: when an additive LoRA (e.g., an
introspection adapter) is later attached on top of gauged weights, its
contribution sits in a different basis than the one it was trained against.
"""
import torch


def build_gauge_block(angles: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Build a head_dim × head_dim block-diagonal gauge in GPT-NeoX RoPE layout.

    Args:
        angles: shape (head_dim // 2,), rotation angle for each RoPE pair.
        head_dim: per-head dimension. Must equal 2 * len(angles).

    Returns:
        G: shape (head_dim, head_dim). For each i in [0, head_dim/2):
           G[i, i]           = cos(angles[i])
           G[i + d/2, i + d/2] = cos(angles[i])
           G[i, i + d/2]     = -sin(angles[i])
           G[i + d/2, i]     = sin(angles[i])
        All other entries are zero. G is orthogonal, det 1 (block-by-block).
    """
    d = head_dim
    half = d // 2
    assert angles.shape == (half,), f"angles shape {angles.shape}, expected ({half},)"
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    G = torch.zeros(d, d, dtype=angles.dtype, device=angles.device)
    c = torch.cos(angles)
    s = torch.sin(angles)
    idx = torch.arange(half, device=angles.device)
    G[idx, idx] = c
    G[idx + half, idx + half] = c
    G[idx, idx + half] = -s
    G[idx + half, idx] = s
    return G


def apply_gauge_to_layer(
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    gauge_angles_per_kv_head: torch.Tensor,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a per-K-head gauge to (W_Q, W_K) respecting GQA grouping.

    The same gauge G_h is applied to:
      - K head h (rows [h*d : (h+1)*d] of W_K)
      - All Q heads in K head h's group (rows of W_Q for Q heads
        [h*group_size : (h+1)*group_size]).

    This shared-G constraint is required for the per-head attention score to be
    gauge-invariant.

    Args:
        W_Q: (n_q_heads * head_dim, hidden) — HF nn.Linear weight convention.
        W_K: (n_kv_heads * head_dim, hidden)
        gauge_angles_per_kv_head: (n_kv_heads, head_dim // 2) — rotation angles
            per K head, per RoPE pair.
        n_q_heads, n_kv_heads, head_dim: model arch constants.

    Returns:
        (W_Q_new, W_K_new) with the gauge applied. Same shapes/dtypes as inputs.
    """
    assert n_q_heads % n_kv_heads == 0, \
        f"n_q_heads ({n_q_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
    assert gauge_angles_per_kv_head.shape == (n_kv_heads, head_dim // 2), \
        f"expected gauge_angles shape ({n_kv_heads}, {head_dim//2}), got {gauge_angles_per_kv_head.shape}"

    group_size = n_q_heads // n_kv_heads
    W_Q_new = W_Q.clone()
    W_K_new = W_K.clone()

    for kv_h in range(n_kv_heads):
        G = build_gauge_block(gauge_angles_per_kv_head[kv_h], head_dim)
        Gt = G.T  # for HF weight convention: W'[head] = G^T @ W[head]
        # All Q heads in this K head's group
        for offset in range(group_size):
            q_h = kv_h * group_size + offset
            r0, r1 = q_h * head_dim, (q_h + 1) * head_dim
            W_Q_new[r0:r1, :] = Gt @ W_Q[r0:r1, :]
        # The K head itself (G^{-T} = G for pure rotation → same form as Q)
        r0, r1 = kv_h * head_dim, (kv_h + 1) * head_dim
        W_K_new[r0:r1, :] = Gt @ W_K[r0:r1, :]

    return W_Q_new, W_K_new


def build_orthogonal_gauge(head_dim: int, generator: torch.Generator | None = None,
                            dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Random orthogonal matrix uniformly distributed on O(head_dim) (Haar measure).

    Constructed by QR-decomposing a random Gaussian matrix and sign-correcting
    R's diagonal so the columns of Q are uniformly distributed on the unit
    sphere — the standard Haar-on-O(d) trick.

    Used for the V/O gauge, where there is no RoPE constraint and any invertible
    G works. Orthogonal is well-conditioned (G^{-1} = G^T).
    """
    A = torch.randn(head_dim, head_dim, generator=generator, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    Q = Q * R.diag().sign()
    return Q


def apply_vo_gauge_to_layer(
    W_V: torch.Tensor,
    W_O: torch.Tensor,
    gauges_per_kv_head: torch.Tensor,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a per-K-head V/O gauge so the attention output is unchanged.

    The math, per K head h with gauge G_h:
      v_h_new       = v_h G_h                  (post-multiply v on head_dim)
      W_O[h]_new    = G_h^{-1} W_O[h]          (compensate at output projection)

    Attention output per head: attn × v_h_new × W_O[h]_new
                             = attn × v_h G_h × G_h^{-1} W_O[h]
                             = attn × v_h × W_O[h]                 ← unchanged

    For orthogonal G_h, G_h^{-1} = G_h^T.

    In HF weight storage:
      W_V.weight is (n_kv*d, hidden); per-V-head rows [h*d:(h+1)*d] act as
        v_h = x @ W_V[h*d:(h+1)*d].T   ⇒  v_h_new = v_h G_h
        ⇒ W_V[h*d:(h+1)*d, :]_new = G_h^T @ W_V[h*d:(h+1)*d, :]

      W_O.weight is (hidden, n_q*d); per-Q-head columns [h*d:(h+1)*d]
        output = concat([attn_h × v_h]_h) @ W_O.weight.T
        ⇒ per-Q-head:  W_O[:, h*d:(h+1)*d]_new = W_O[:, h*d:(h+1)*d] @ G_h

    GQA constraint: V has one head per KV head, but O has one column-block per
    Q head. The Q heads in a KV head's group share the same V, so the same G_h
    must be applied to all of them at the O projection — otherwise the v_h G_h
    arriving at O head q gets multiplied by a different G_q^{-1} and doesn't
    cancel.

    Args:
        W_V: (n_kv_heads * head_dim, hidden) — HF nn.Linear weight convention.
        W_O: (hidden, n_q_heads * head_dim)
        gauges_per_kv_head: (n_kv_heads, head_dim, head_dim) — one full
            invertible matrix per K head.
        n_q_heads, n_kv_heads, head_dim: model arch constants.

    Returns:
        (W_V_new, W_O_new) with the gauge applied. Same shapes/dtypes as inputs.
    """
    assert n_q_heads % n_kv_heads == 0, \
        f"n_q_heads ({n_q_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
    assert gauges_per_kv_head.shape == (n_kv_heads, head_dim, head_dim), \
        f"expected gauges shape ({n_kv_heads}, {head_dim}, {head_dim}), got {gauges_per_kv_head.shape}"

    group_size = n_q_heads // n_kv_heads
    W_V_new = W_V.clone()
    W_O_new = W_O.clone()

    for kv_h in range(n_kv_heads):
        G = gauges_per_kv_head[kv_h]
        Gt = G.T
        # V: rows [h*d:(h+1)*d] of W_V become Gt @ rows
        r0, r1 = kv_h * head_dim, (kv_h + 1) * head_dim
        W_V_new[r0:r1, :] = Gt @ W_V[r0:r1, :]
        # O: column-block [q*d:(q+1)*d] of W_O becomes block @ G, for every
        # Q head q in this KV head's group
        for offset in range(group_size):
            q_h = kv_h * group_size + offset
            c0, c1 = q_h * head_dim, (q_h + 1) * head_dim
            W_O_new[:, c0:c1] = W_O[:, c0:c1] @ G

    return W_V_new, W_O_new


def build_random_permutation(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random permutation of {0, ..., n-1}, uniform over S_n.

    Returns an int64 tensor of shape (n,) where perm[i] = π(i).
    Use as a fancy-index to permute rows/columns of a tensor.
    """
    return torch.randperm(n, generator=generator, dtype=torch.int64)


def apply_mlp_permutation_to_layer(
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    perm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a hidden-dim permutation to a SiLU-gated MLP, preserving output.

    For an MLP computing `down(silu(gate(x)) * up(x))`:

      W_gate' = P @ W_gate           ← permute output rows of gate
      W_up'   = P @ W_up             ← permute output rows of up (SAME π)
      W_down' = W_down @ P^T         ← permute input cols of down (inverse)

    Then for any x:
      gate'(x)[i]      = gate(x)[π(i)]
      up'(x)[i]        = up(x)[π(i)]
      (silu(gate') * up')[i]   = silu(gate(x)[π(i)]) * up(x)[π(i)]
      down'(silu * up)[j]      = Σ_i W_down[j, π(i)] · silu(gate(x)[π(i)]) · up(x)[π(i)]
                                = Σ_k W_down[j, k]   · silu(gate(x)[k])    · up(x)[k]
                                = down(silu(gate(x)) * up(x))[j]   ← unchanged

    The same π MUST be applied to W_gate and W_up (the elementwise product
    requires aligned slots). down uses the inverse permutation on columns
    to undo.

    Args:
        W_gate, W_up:   (intermediate, hidden) — HF Linear weight convention.
        W_down:         (hidden, intermediate)
        perm:           (intermediate,) int64 — π[i] is the index taken from
                        the original to land in slot i of the permuted.

    Returns:
        (W_gate_new, W_up_new, W_down_new) — same shapes/dtypes.
    """
    intermediate = W_gate.shape[0]
    assert W_up.shape[0] == intermediate, f"W_up rows {W_up.shape[0]} != W_gate rows {intermediate}"
    assert W_down.shape[1] == intermediate, f"W_down cols {W_down.shape[1]} != intermediate {intermediate}"
    assert perm.shape == (intermediate,), f"perm shape {perm.shape}, expected ({intermediate},)"

    # Row-permute output dim of gate and up
    W_gate_new = W_gate[perm, :].contiguous()
    W_up_new = W_up[perm, :].contiguous()
    # Column-permute input dim of down by the SAME perm
    W_down_new = W_down[:, perm].contiguous()
    return W_gate_new, W_up_new, W_down_new
