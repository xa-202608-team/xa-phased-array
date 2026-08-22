"""模型唯一构造函数 factory (F1 收口)。

run_groups/导出器/predict_gru 三处共用 build_transfer_model，消除 GRU hidden/layers/
dropout 训练-导出-推理三处漂移。GRU 架构参数 (hidden=64/layers=2/dropout=0.1) 现从
config.model.gru 显式读取 (F1-A: 不再吃 adapter 隐式默认); factory 必须与
run_groups._build_model 完全同构 (后者唯一实现即本 factory)。
"""
import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config                              # noqa: E402
from src.models.factory import build_transfer_model            # noqa: E402

PA_CONFIG = ROOT / "configs" / "phased_array.yaml"


def _cfg():
    return load_config(PA_CONFIG)


def test_factory_builds_gru_main_model():
    """显式 gru 构建 → GRU 主模型, 黑盒结构符合 TransferModel.forward 契约。"""
    cfg = _cfg()
    model = build_transfer_model(cfg, n_features=4, n_target=4, encoder_type="gru")
    assert type(model).__name__ == "TransferModel"
    assert model.encoder.__class__.__name__ == "GRUEncoder"

    # forward: (B=2, L=64, n=4) → hi(B) rul(B) z(B,latent)
    x = torch.randn(2, 64, 4)
    hi, rul, z = model(x)
    assert hi.shape == (2,) and rul.shape == (2,) and z.shape == (2, cfg["model"]["latent_dim"])
    assert (hi >= 0).all() and (hi <= 1).all(), "hi = sigmoid 输出应在 [0,1]"
    assert (rul >= 0).all(), "rul = softplus 应 ≥ 0"


def test_factory_gru_hidden_layers_match_adapter_default():
    """GRU 必须用 config.model.gru 默认 64/2 (当前 config 值恰等于 adapter 默认, 防漂移)。

    注意: 该 64/2 现来自 config (factory 显式读取), 不再是 adapter 隐式默认 —
    值相等只是当前 config 恰好 64/2。config 驱动性见 test_factory_reads_gru_from_config_deepcopy。
    """
    cfg = _cfg()
    model = build_transfer_model(cfg, n_features=4, n_target=4, encoder_type="gru")
    gru = model.encoder.gru  # nn.GRU(4,64,2,batch_first=True)
    assert gru.hidden_size == 64, "config.model.gru.hidden=64 (当前默认)"
    assert gru.num_layers == 2, "config.model.gru.num_layers=2 (当前默认)"
    assert gru.input_size == 4, f"n_features 应=4, 实={gru.input_size}"


def test_factory_reads_gru_from_config_deepcopy():
    """factory 从 config.model.gru 显式读取 (非硬编码 64/2): 改 cfg 后模型跟着变。

    用 copy.deepcopy 改 hidden=128/num_layers=1 再构建, 断言 encoder.gru 跟随 —
    证明 factory 读 config 而非写死。deepcopy 避免污染本模块共享的 _cfg() 结果。
    """
    cfg = copy.deepcopy(_cfg())
    cfg["model"]["gru"]["hidden"] = 128
    cfg["model"]["gru"]["num_layers"] = 1
    model = build_transfer_model(cfg, n_features=4, n_target=4, encoder_type="gru")
    gru = model.encoder.gru
    assert gru.hidden_size == 128, "factory 应从 config.model.gru.hidden 读取 (改 128 后应跟随)"
    assert gru.num_layers == 1, "factory 应从 config.model.gru.num_layers 读取 (改 1 后应跟随)"
    assert gru.input_size == 4


def test_factory_roundtrip_weights_with_run_groups_model():
    """factory 构建的模型与 run_groups._build_model 权重形状一致 (同 config 同 seed 可互载)。"""
    cfg = _cfg()
    import src.experiments.run_groups as rg
    m_rg = rg._build_model(cfg, n_features=4, n_target=4, device="cpu", encoder_override="gru")
    m_fac = build_transfer_model(cfg, n_features=4, n_target=4, encoder_type="gru")
    for (k1, p1), (k2, p2) in zip(m_rg.state_dict().items(), m_fac.state_dict().items()):
        assert k1 == k2, f"键不一致: {k1} vs {k2}"
        assert p1.shape == p2.shape, f"{k1} 形状不同: {p1.shape} vs {p2.shape}"
    # 同 init (seed) 结果应精确一致 → factory 与 _build_model 同构
    from src.utils import set_seed
    x = torch.linspace(-0.5, 0.5, 64 * 4).reshape(1, 64, 4)   # 固定输入, 不依赖 RNG
    set_seed(42, True, False)
    m_rg2 = rg._build_model(cfg, n_features=4, n_target=4, device="cpu", encoder_override="gru")
    set_seed(42, True, False)
    m_fac2 = build_transfer_model(cfg, n_features=4, n_target=4, encoder_type="gru")
    m_rg2.eval()   # eval 模式排除 dropout 随机性, 比的是架构同构
    m_fac2.eval()
    with torch.no_grad():
        out_rg = m_rg2(x)[1]
        out_fac = m_fac2(x)[1]
    assert torch.allclose(out_rg, out_fac, atol=1e-6), "同 seed eval 下 factory 与 _build_model 输出应一致"


def test_factory_tcn_and_override():
    """encoder_type 覆盖与 TCN 分支。"""
    cfg = _cfg()
    m_tcn = build_transfer_model(cfg, n_features=4, n_target=4, encoder_type="tcn")
    assert m_tcn.encoder.__class__.__name__ == "TCNEncoder"
    m_gru = build_transfer_model(cfg, n_features=4, n_target=4, encoder_type="gru")
    assert m_gru.encoder.__class__.__name__ == "GRUEncoder"