"""A1 α-soft transfer 契约测试 (§4d 四审方案).

θ₀ = θ_rand + α·(θ_src − θ_rand) 的三条硬约束:
  ① 端点数学: α=0 → state_dict 逐位不变; α=1 → 严格等价 source 臂 load_pretrained;
     α=0.5 → 逐元素中点
  ② 插值范围 = encoder.* 浮点张量 (与 load_pretrained 加载范围一致,
     adapter/heads 保持 θ_rand — α=1 端点可比性不被插值范围污染)
  ③ RNG 同源: _build_model 调用点在 α-soft 分支之前, 且 alpha_soft_finetune
     不走 load_pretrained 分支 → θ_rand 与 ch_random_full_finetune 同 seed 同源
     (α=0 校验臂应逐位复现 random 臂; torch.load/load_state_dict 不消耗 RNG)
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.run_groups import (                          # noqa: E402
    _GROUP_MAP,
    _alpha_from_group,
    _interpolate_encoder,
    _source_ckpt_name,
)
from src.transfer.adapter import TransferModel                   # noqa: E402


def _mk_model(seed=0):
    torch.manual_seed(seed)
    return TransferModel(encoder_type="gru", n_features=4, n_target=4,
                         latent_dim=8, adapter_hidden=8)


def _mk_src_sd(shift=1.0):
    """伪造源 ckpt state_dict: encoder.* 全部偏移 shift, 非 encoder 键也带上 (应被忽略)。"""
    m = _mk_model(seed=123)
    sd = m.state_dict()
    fake = {}
    for k, v in sd.items():
        fake[k] = v + shift if k.startswith("encoder.") else v + 99.0   # 非 encoder 键故意污染
    return {"model": fake}


class TestAlphaFromGroup:
    def test_parses_alpha_values(self):
        assert _alpha_from_group("ch_alpha_soft_a005") == 0.05
        assert _alpha_from_group("ch_alpha_soft_a050") == 0.50
        assert _alpha_from_group("ch_alpha_soft_a000") == 0.0

    def test_parses_with_kshot_suffix(self):
        # main() 传给 run_one_group 的 group_name 带 _k{shot} 后缀 (tag_name)
        assert _alpha_from_group("ch_alpha_soft_a005_k3") == 0.05
        assert _alpha_from_group("ch_alpha_soft_a025_k3") == 0.25

    def test_non_alpha_groups_return_none(self):
        assert _alpha_from_group("ch_source_mmd_physics_k3") is None
        assert _alpha_from_group("ch_random_full_finetune_k3") is None
        assert _alpha_from_group(None) is None
        assert _alpha_from_group("ch_source_igbt") is None


class TestInterpolateEncoder:
    def test_alpha_zero_is_identity(self):
        model = _mk_model(seed=7)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        sd = _mk_src_sd()["model"]
        n = _interpolate_encoder(model, sd, alpha=0.0)
        after = model.state_dict()
        assert n > 0
        assert set(before) == set(after)
        for k in before:
            assert torch.equal(before[k], after[k]), f"α=0 改动了 {k}"

    def test_alpha_one_equals_load_pretrained(self):
        # α=1 端点契约: 与 source 臂 load_pretrained 同一 state_dict
        m_a = _mk_model(seed=7)                       # α-soft 路径
        m_b = _mk_model(seed=7)                       # source 臂路径 (同 seed → 同 θ_rand)
        sd_full = _mk_src_sd()["model"]
        sd_enc = {k: v for k, v in sd_full.items() if k.startswith("encoder.")}
        _interpolate_encoder(m_a, sd_enc, alpha=1.0)
        m_b.load_state_dict(sd_enc, strict=False)     # = load_pretrained 的内核逻辑
        sa, sb = m_a.state_dict(), m_b.state_dict()
        assert set(sa) == set(sb)
        for k in sa:
            assert torch.allclose(sa[k], sb[k], atol=1e-6), f"α=1 端点在 {k} 上不等价"

    def test_alpha_half_is_elementwise_midpoint(self):
        m_rand = _mk_model(seed=7)
        m_src = _mk_model(seed=123)
        sd_enc = {k: v for k, v in m_src.state_dict().items() if k.startswith("encoder.")}
        m_mix = _mk_model(seed=7)
        _interpolate_encoder(m_mix, sd_enc, alpha=0.5)
        sa, sm = m_rand.state_dict(), m_mix.state_dict()
        for k in sa:
            if k in sd_enc and sa[k].dtype.is_floating_point:
                expect = sa[k] + 0.5 * (sd_enc[k] - sa[k])
                assert torch.allclose(sm[k], expect, atol=1e-6), f"{k} 非中点"

    def test_non_encoder_keys_untouched(self):
        # 插值范围契约: adapter/heads 保持 θ_rand (源 ckpt 中被污染的 99.0 不得渗入)
        model = _mk_model(seed=7)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        sd_full = _mk_src_sd()["model"]               # 非 encoder 键 = 原值+99 (污染探针)
        sd_enc = {k: v for k, v in sd_full.items() if k.startswith("encoder.")}
        _interpolate_encoder(model, sd_enc, alpha=0.5)
        after = model.state_dict()
        n_enc = sum(1 for k in before if k in sd_enc and before[k].dtype.is_floating_point)
        changed = [k for k in before if not torch.equal(before[k], after[k])]
        assert set(changed) <= set(sd_enc), f"插值越界: {set(changed) - set(sd_enc)}"
        assert len(changed) == n_enc, "encoder 浮点张量应全部被插值"


class TestGroupRegistration:
    def test_alpha_groups_registered_with_gru(self):
        # 架构纪律 (清零重审): 迁移归因组统一显式 GRU
        for g, a in [("ch_alpha_soft_a000", 0.0), ("ch_alpha_soft_a005", 0.05),
                     ("ch_alpha_soft_a010", 0.10), ("ch_alpha_soft_a025", 0.25),
                     ("ch_alpha_soft_a050", 0.50)]:
            mode, enc = _GROUP_MAP[g]
            assert mode == "alpha_soft_finetune", g
            assert enc == "gru", g
            assert _alpha_from_group(g) == a

    def test_alpha_mode_excluded_from_direct_load(self):
        # alpha_soft 不得走 load_pretrained 全量加载分支 (那等于 α=1)
        src = (ROOT / "src" / "experiments" / "run_groups.py").read_text(encoding="utf-8")
        needle = 'if mode not in ("target_only", "random_frozen", "random_full_mmd", "random_full_nommd",'
        assert needle in src, "mode 白名单被改动, alpha_soft_finetune 需在排除名单内"
        idx_block = src.index(needle)
        block = src[idx_block:idx_block + 200]
        assert "alpha_soft_finetune" in block, "alpha_soft_finetune 不在 load_pretrained 排除名单"

    def test_build_model_precedes_alpha_branch(self):
        # θ_rand 同源的代码结构前提: _build_model (确定 θ_rand) 在 α 插值分支之前
        src = (ROOT / "src" / "experiments" / "run_groups.py").read_text(encoding="utf-8")
        fn = src[src.index("def run_one_group"):src.index("def _paired_delta_ci")]
        i_build = fn.index("model = _build_model")
        i_alpha = fn.index("_alpha = _alpha_from_group")
        i_interp = fn.index("_interpolate_encoder(model")
        assert i_build < i_alpha < i_interp, "α 插值必须在 _build_model 之后 (θ_rand 先确定)"

    def test_source_ckpt_name_shared_with_source_arm(self):
        # α=1 端点可比性: alpha 组与 MOSFET source 臂解析出同一 ckpt 文件名
        assert (_source_ckpt_name("ch_alpha_soft_a005_k3", "phased_array", "gru")
                == _source_ckpt_name("ch_source_mmd_physics_k3", "phased_array", "gru"))
        # §4c 源域 tag 约定不回归
        assert "igbt" in _source_ckpt_name("ch_source_igbt_k3", "phased_array", "gru")
        assert "mosfet_igbt" in _source_ckpt_name("ch_source_multi_k3", "phased_array", "gru")
