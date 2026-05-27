#!/usr/bin/env python3
"""CPU-only sanity tests for gauge_lib. Run: uv run python scripts/experiments/gauge_math_sanity.py
Exits 0 on success, 1 on failure."""
import math
import sys

import torch

from gauge_lib import build_gauge_block


def test_build_gauge_block_shape_and_structure():
    """G has shape (d, d) and is block-diagonal in NeoX layout: pairs (i, i+d/2)."""
    d = 8
    half = d // 2
    angles = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    G = build_gauge_block(angles, d)
    assert G.shape == (d, d), f"expected shape ({d},{d}), got {G.shape}"
    # Only the four "block" entries per pair should be non-zero
    nonzero_expected = set()
    for i in range(half):
        nonzero_expected.update([(i, i), (i, i + half), (i + half, i), (i + half, i + half)])
    nonzero_actual = set((int(r), int(c)) for r, c in (G != 0).nonzero())
    assert nonzero_actual == nonzero_expected, \
        f"unexpected nonzero pattern: extra={nonzero_actual - nonzero_expected}, missing={nonzero_expected - nonzero_actual}"
    # Determinant of each 2D block should be 1 (pure rotation)
    for i in range(half):
        c, s = G[i, i], G[i, i + half]
        c2, s2 = G[i + half, i + half], G[i + half, i]
        assert torch.isclose(c, c2), f"pair {i}: cos values differ"
        assert torch.isclose(s + s2, torch.tensor(0.0, dtype=torch.float64)), f"pair {i}: sin signs wrong"
        det = c * c2 - (G[i, i + half] * G[i + half, i])
        assert torch.isclose(det, torch.tensor(1.0, dtype=torch.float64)), f"pair {i}: det={det.item()}, expected 1"


def _apply_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Mirror HF's apply_rotary_pos_emb for a single tensor.
    x: (..., head_dim). cos, sin: (..., head_dim).
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def test_gauge_commutes_with_neox_rope():
    """For any per-pair RoPE angles theta and any gauge angles phi, applying
    RoPE then gauge equals applying gauge then RoPE. This is the load-bearing
    invariance the attack relies on."""
    d = 8
    half = d // 2
    torch.manual_seed(0)

    # Random query vector in head_dim space
    q = torch.randn(d, dtype=torch.float64)

    # Random RoPE angles (per pair)
    theta = torch.rand(half, dtype=torch.float64) * 2 * math.pi
    # NeoX cos/sin tensors duplicate values across the half-split
    cos = torch.cat([torch.cos(theta), torch.cos(theta)], dim=-1)
    sin = torch.cat([torch.sin(theta), torch.sin(theta)], dim=-1)

    # Random gauge
    phi = torch.rand(half, dtype=torch.float64) * 2 * math.pi
    G = build_gauge_block(phi, d)

    # Order A: gauge first, then RoPE
    q_GR = _apply_rope_neox(q @ G, cos, sin)
    # Order B: RoPE first, then gauge
    q_RG = _apply_rope_neox(q, cos, sin) @ G

    diff = (q_GR - q_RG).abs().max().item()
    assert diff < 1e-10, f"gauge does not commute with NeoX RoPE: max diff {diff:.2e}"


def _attn_score(W_Q, W_K, x, n_q_heads, n_kv_heads, head_dim, cos, sin):
    """Toy attention score (per-head, before softmax), matching HF Llama math.

    W_Q: (n_q_heads * head_dim, hidden)
    W_K: (n_kv_heads * head_dim, hidden)
    x:   (seq, hidden)
    cos, sin: (seq, head_dim) — same for all heads
    Returns: (n_q_heads, seq, seq) — QK^T per Q head, K repeated across the GQA group.
    """
    seq, hidden = x.shape
    q = x @ W_Q.T  # (seq, n_q * d)
    k = x @ W_K.T  # (seq, n_kv * d)
    q = q.view(seq, n_q_heads, head_dim)
    k = k.view(seq, n_kv_heads, head_dim)
    # Apply RoPE per-head (cos/sin broadcast over heads)
    cos_b = cos.unsqueeze(1)  # (seq, 1, d)
    sin_b = sin.unsqueeze(1)
    q = _apply_rope_neox(q, cos_b, sin_b)
    k = _apply_rope_neox(k, cos_b, sin_b)
    # Expand K across the GQA group: each K head shared by group_size Q heads
    group_size = n_q_heads // n_kv_heads
    k_expanded = k.repeat_interleave(group_size, dim=1)  # (seq, n_q, d)
    # Per-head QK^T
    q_p = q.permute(1, 0, 2)            # (n_q, seq, d)
    k_p = k_expanded.permute(1, 0, 2)   # (n_q, seq, d)
    return q_p @ k_p.transpose(-1, -2)   # (n_q, seq, seq)


def test_apply_gauge_to_layer_preserves_attention_score():
    """End-to-end: per-K-head gauge (shared across GQA group) leaves the
    per-head attention scores unchanged."""
    from gauge_lib import apply_gauge_to_layer
    torch.manual_seed(1)

    n_q_heads = 4
    n_kv_heads = 2     # GQA group size 2
    head_dim = 8
    hidden = 16
    seq = 3
    half = head_dim // 2

    W_Q = torch.randn(n_q_heads * head_dim, hidden, dtype=torch.float64)
    W_K = torch.randn(n_kv_heads * head_dim, hidden, dtype=torch.float64)

    # Per-K-head gauge angles
    gauge_angles = torch.rand(n_kv_heads, half, dtype=torch.float64) * 2 * math.pi

    # RoPE
    theta = torch.rand(seq, half, dtype=torch.float64) * 2 * math.pi
    cos = torch.cat([torch.cos(theta), torch.cos(theta)], dim=-1)
    sin = torch.cat([torch.sin(theta), torch.sin(theta)], dim=-1)
    x = torch.randn(seq, hidden, dtype=torch.float64)

    score_before = _attn_score(W_Q, W_K, x, n_q_heads, n_kv_heads, head_dim, cos, sin)

    W_Q_new, W_K_new = apply_gauge_to_layer(
        W_Q, W_K, gauge_angles, n_q_heads, n_kv_heads, head_dim
    )
    score_after = _attn_score(W_Q_new, W_K_new, x, n_q_heads, n_kv_heads, head_dim, cos, sin)

    diff = (score_before - score_after).abs().max().item()
    assert diff < 1e-9, f"gauge changes attention score: max diff {diff:.2e}"


def test_apply_gauge_to_layer_breaks_score_if_gqa_grouping_wrong():
    """Sanity: if we use a *different* gauge for each Q head in a group
    (instead of sharing the K head's gauge), the score should change.
    This guards against accidentally treating Q heads as independent."""
    from gauge_lib import apply_gauge_to_layer, build_gauge_block
    torch.manual_seed(2)
    n_q_heads = 4
    n_kv_heads = 2
    head_dim = 8
    hidden = 16
    seq = 3
    half = head_dim // 2

    W_Q = torch.randn(n_q_heads * head_dim, hidden, dtype=torch.float64)
    W_K = torch.randn(n_kv_heads * head_dim, hidden, dtype=torch.float64)
    theta = torch.rand(seq, half, dtype=torch.float64) * 2 * math.pi
    cos = torch.cat([torch.cos(theta), torch.cos(theta)], dim=-1)
    sin = torch.cat([torch.sin(theta), torch.sin(theta)], dim=-1)
    x = torch.randn(seq, hidden, dtype=torch.float64)

    score_before = _attn_score(W_Q, W_K, x, n_q_heads, n_kv_heads, head_dim, cos, sin)

    # Use a per-Q-head gauge (wrong) — sharing only within the K head should hold.
    # Construct: same K-head gauge, but use a *different* random angle on the second Q head of each group.
    bad_q_angles = torch.rand(n_q_heads, half, dtype=torch.float64) * 2 * math.pi
    kv_angles = bad_q_angles[::2].clone()   # first Q in each group
    # Force Q heads 1, 3 (second in each group) to use a different gauge:
    bad_q_angles[1] = bad_q_angles[1] + 0.5
    bad_q_angles[3] = bad_q_angles[3] + 0.5

    # Manually apply: each Q head uses its own gauge, K head uses kv_angles
    W_Q_bad = W_Q.clone()
    W_K_bad = W_K.clone()
    for q_h in range(n_q_heads):
        Gq = build_gauge_block(bad_q_angles[q_h], head_dim)
        W_Q_bad[q_h*head_dim:(q_h+1)*head_dim, :] = Gq.T @ W_Q[q_h*head_dim:(q_h+1)*head_dim, :]
    for kv_h in range(n_kv_heads):
        Gk = build_gauge_block(kv_angles[kv_h], head_dim)
        W_K_bad[kv_h*head_dim:(kv_h+1)*head_dim, :] = Gk.T @ W_K[kv_h*head_dim:(kv_h+1)*head_dim, :]

    score_bad = _attn_score(W_Q_bad, W_K_bad, x, n_q_heads, n_kv_heads, head_dim, cos, sin)
    diff = (score_before - score_bad).abs().max().item()
    assert diff > 1e-3, \
        f"score did not change under wrong per-Q-head gauge — test is not exercising GQA grouping (diff {diff:.2e})"


def _full_attn_output(W_Q, W_K, W_V, W_O, x, n_q_heads, n_kv_heads, head_dim, cos, sin):
    """Compute the full attention block output (post-softmax, post-W_O),
    matching HF Llama math: q,k,v projections → RoPE on q,k → softmax(QK^T/sqrt(d)) → @v → concat → @ W_O.

    Shapes:
      W_Q: (n_q*d, hidden)   W_K: (n_kv*d, hidden)
      W_V: (n_kv*d, hidden)  W_O: (hidden, n_q*d)
      x:   (seq, hidden)     cos,sin: (seq, d)
    Returns: (seq, hidden)
    """
    seq, hidden = x.shape
    q = (x @ W_Q.T).view(seq, n_q_heads, head_dim)
    k = (x @ W_K.T).view(seq, n_kv_heads, head_dim)
    v = (x @ W_V.T).view(seq, n_kv_heads, head_dim)

    cos_b = cos.unsqueeze(1); sin_b = sin.unsqueeze(1)
    q = _apply_rope_neox(q, cos_b, sin_b)
    k = _apply_rope_neox(k, cos_b, sin_b)

    group_size = n_q_heads // n_kv_heads
    k_e = k.repeat_interleave(group_size, dim=1)   # (seq, n_q, d)
    v_e = v.repeat_interleave(group_size, dim=1)   # (seq, n_q, d)

    q_p = q.permute(1, 0, 2)              # (n_q, seq, d)
    k_p = k_e.permute(1, 0, 2)            # (n_q, seq, d)
    v_p = v_e.permute(1, 0, 2)            # (n_q, seq, d)
    scores = (q_p @ k_p.transpose(-1, -2)) / (head_dim ** 0.5)  # (n_q, seq, seq)
    attn = torch.softmax(scores, dim=-1)
    out_per_head = attn @ v_p             # (n_q, seq, d)
    concat = out_per_head.permute(1, 0, 2).reshape(seq, n_q_heads * head_dim)  # (seq, n_q*d)
    out = concat @ W_O.T                  # (seq, hidden)
    return out


def test_apply_vo_gauge_preserves_attention_output():
    """V/O gauge with shared-per-K-head sharing leaves the FULL attention block
    output unchanged (modulo float64 noise). Test target is the post-W_O output,
    not just the score, since V/O participates in everything downstream of QK^T."""
    from gauge_lib import apply_vo_gauge_to_layer, build_orthogonal_gauge
    torch.manual_seed(3)

    n_q_heads = 4
    n_kv_heads = 2     # GQA group size 2
    head_dim = 8
    hidden = 16
    seq = 3
    half = head_dim // 2

    W_Q = torch.randn(n_q_heads * head_dim, hidden, dtype=torch.float64)
    W_K = torch.randn(n_kv_heads * head_dim, hidden, dtype=torch.float64)
    W_V = torch.randn(n_kv_heads * head_dim, hidden, dtype=torch.float64)
    W_O = torch.randn(hidden, n_q_heads * head_dim, dtype=torch.float64)

    theta = torch.rand(seq, half, dtype=torch.float64) * 2 * math.pi
    cos = torch.cat([torch.cos(theta), torch.cos(theta)], dim=-1)
    sin = torch.cat([torch.sin(theta), torch.sin(theta)], dim=-1)
    x = torch.randn(seq, hidden, dtype=torch.float64)

    gen = torch.Generator().manual_seed(31)
    gauges = torch.stack([build_orthogonal_gauge(head_dim, generator=gen) for _ in range(n_kv_heads)])

    out_before = _full_attn_output(W_Q, W_K, W_V, W_O, x, n_q_heads, n_kv_heads, head_dim, cos, sin)
    W_V_new, W_O_new = apply_vo_gauge_to_layer(W_V, W_O, gauges, n_q_heads, n_kv_heads, head_dim)
    out_after = _full_attn_output(W_Q, W_K, W_V_new, W_O_new, x, n_q_heads, n_kv_heads, head_dim, cos, sin)

    diff = (out_before - out_after).abs().max().item()
    assert diff < 1e-9, f"V/O gauge changes attention output: max diff {diff:.2e}"


def test_apply_vo_gauge_breaks_output_if_o_not_compensated():
    """Sanity: if we apply gauge to W_V but DON'T compensate at W_O, the
    attention output should change. Guards against accidentally writing a
    no-op that always passes."""
    from gauge_lib import build_orthogonal_gauge
    torch.manual_seed(4)

    n_q_heads = 4
    n_kv_heads = 2
    head_dim = 8
    hidden = 16
    seq = 3
    half = head_dim // 2

    W_Q = torch.randn(n_q_heads * head_dim, hidden, dtype=torch.float64)
    W_K = torch.randn(n_kv_heads * head_dim, hidden, dtype=torch.float64)
    W_V = torch.randn(n_kv_heads * head_dim, hidden, dtype=torch.float64)
    W_O = torch.randn(hidden, n_q_heads * head_dim, dtype=torch.float64)

    theta = torch.rand(seq, half, dtype=torch.float64) * 2 * math.pi
    cos = torch.cat([torch.cos(theta), torch.cos(theta)], dim=-1)
    sin = torch.cat([torch.sin(theta), torch.sin(theta)], dim=-1)
    x = torch.randn(seq, hidden, dtype=torch.float64)

    out_before = _full_attn_output(W_Q, W_K, W_V, W_O, x, n_q_heads, n_kv_heads, head_dim, cos, sin)

    gen = torch.Generator().manual_seed(41)
    gauges = torch.stack([build_orthogonal_gauge(head_dim, generator=gen) for _ in range(n_kv_heads)])
    # Apply gauge to V only — DON'T touch W_O. Output should change.
    W_V_bad = W_V.clone()
    for kv_h in range(n_kv_heads):
        r0, r1 = kv_h * head_dim, (kv_h + 1) * head_dim
        W_V_bad[r0:r1, :] = gauges[kv_h].T @ W_V[r0:r1, :]

    out_bad = _full_attn_output(W_Q, W_K, W_V_bad, W_O, x, n_q_heads, n_kv_heads, head_dim, cos, sin)
    diff = (out_before - out_bad).abs().max().item()
    assert diff > 1e-3, \
        f"output didn't change when V was gauged without O compensation — test isn't exercising V/O coupling (diff {diff:.2e})"


def _mlp_silu_glu(W_gate, W_up, W_down, x):
    """SiLU-gated MLP: down(silu(gate(x)) * up(x)).
    W_gate, W_up: (intermediate, hidden); W_down: (hidden, intermediate); x: (..., hidden).
    """
    silu = torch.nn.functional.silu
    g = x @ W_gate.T
    u = x @ W_up.T
    z = silu(g) * u
    return z @ W_down.T


def test_apply_mlp_permutation_preserves_output():
    """Same π applied to W_up/W_gate rows + W_down cols leaves MLP output unchanged."""
    from gauge_lib import apply_mlp_permutation_to_layer, build_random_permutation
    torch.manual_seed(5)
    hidden = 16
    intermediate = 40
    seq = 3
    x = torch.randn(seq, hidden, dtype=torch.float64)
    W_gate = torch.randn(intermediate, hidden, dtype=torch.float64)
    W_up = torch.randn(intermediate, hidden, dtype=torch.float64)
    W_down = torch.randn(hidden, intermediate, dtype=torch.float64)

    gen = torch.Generator().manual_seed(51)
    perm = build_random_permutation(intermediate, generator=gen)

    out_before = _mlp_silu_glu(W_gate, W_up, W_down, x)
    W_gate_n, W_up_n, W_down_n = apply_mlp_permutation_to_layer(W_gate, W_up, W_down, perm)
    out_after = _mlp_silu_glu(W_gate_n, W_up_n, W_down_n, x)
    diff = (out_before - out_after).abs().max().item()
    assert diff < 1e-9, f"MLP permutation changes output: max diff {diff:.2e}"


def test_mlp_permutation_breaks_output_if_only_up_permuted():
    """Permuting W_up rows without also permuting W_gate AND W_down must change
    the MLP output. Guards against a no-op pass."""
    from gauge_lib import build_random_permutation
    torch.manual_seed(6)
    hidden = 16
    intermediate = 40
    seq = 3
    x = torch.randn(seq, hidden, dtype=torch.float64)
    W_gate = torch.randn(intermediate, hidden, dtype=torch.float64)
    W_up = torch.randn(intermediate, hidden, dtype=torch.float64)
    W_down = torch.randn(hidden, intermediate, dtype=torch.float64)

    gen = torch.Generator().manual_seed(61)
    perm = build_random_permutation(intermediate, generator=gen)
    # Apply perm to W_up only, leave gate and down alone
    W_up_bad = W_up[perm, :].contiguous()
    out_before = _mlp_silu_glu(W_gate, W_up, W_down, x)
    out_bad = _mlp_silu_glu(W_gate, W_up_bad, W_down, x)
    diff = (out_before - out_bad).abs().max().item()
    assert diff > 1e-3, \
        f"output didn't change when only W_up was permuted — test isn't exercising MLP coupling (diff {diff:.2e})"


def main():
    tests = [
        test_build_gauge_block_shape_and_structure,
        test_gauge_commutes_with_neox_rope,
        test_apply_gauge_to_layer_preserves_attention_score,
        test_apply_gauge_to_layer_breaks_score_if_gqa_grouping_wrong,
        test_apply_vo_gauge_preserves_attention_output,
        test_apply_vo_gauge_breaks_output_if_o_not_compensated,
        test_apply_mlp_permutation_preserves_output,
        test_mlp_permutation_breaks_output_if_only_up_permuted,
    ]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed.append(t.__name__)
    if failed:
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
