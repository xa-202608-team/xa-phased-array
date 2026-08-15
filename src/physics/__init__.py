"""src/physics — 确定性物理孪生层 (T4/M4, 零可学参数)。

由模型预测的子阵损伤前推阵列服务性能与服务寿命分布, 与学习层解耦:
  reconstruct_elements → element_rf → array_metrics → service_eol  (一致性: 复现仿真)
  rollout_mc                                          (前推: 物理知情外推服务寿命分布)

全部 twin 量 (c_elem/eta_R/eta_phi/dropout_thr/grad_dir) 不进模型输入 x, 仅此层使用。
"""
