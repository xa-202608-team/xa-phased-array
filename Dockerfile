# =====================================================================
# 相控阵组件 - GPU 复现镜像 (Python 3.12 + CUDA)
#   构建时验证: 导入检查 + 完整 pytest (依赖缺失的测试自带 skip)
#   运行时验证: docker compose run --rm phased_array verify
#
# GPU 支持:
#   - 宿主机需装 NVIDIA Container Toolkit (nvidia-ctk)
#   - docker-compose.yml 已配 deploy.resources.reservations.devices
#   - 无 GPU 环境自动 fallback CPU (代码内 torch.cuda.is_available())
# =====================================================================
FROM python:3.12-slim

ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) 先装依赖 (GPU 版 torch + 其他依赖)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --default-timeout=300 --retries=5 torch==2.11.0 \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      --trusted-host pypi.tuna.tsinghua.edu.cn \
      --trusted-host download.pytorch.org \
 && pip install --no-cache-dir --default-timeout=300 --retries=5 -r requirements.txt \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      --trusted-host pypi.tuna.tsinghua.edu.cn

# 2) 非 root 用户
RUN useradd -m -u 1000 appuser

# 3) 拷贝代码与固定数据 (COPY --chown 设属主, 避免 RUN chown -R 产生 ~6GB 重复层)
#    data/ 仅含 mosfet_canonical.h5 (876KB 源域); simulated/ + features/ 由 entrypoint 再生
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser configs/ /app/configs/
COPY --chown=appuser:appuser tests/ /app/tests/
COPY --chown=appuser:appuser checkpoints/ /app/checkpoints/
COPY --chown=appuser:appuser data/ /app/data/
COPY --chown=appuser:appuser pytest.ini /app/
COPY --chown=appuser:appuser scripts/ /app/scripts/
COPY --chown=appuser:appuser scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && mkdir -p /app/docs && chown appuser:appuser /app/docs

USER appuser

ENV PYTHONPATH=/app

# 4) 构建时验证: 导入检查 + pytest (缺失外部数据的测试自带 skip, 不失败)
RUN python -c "\
import src.sim.phased_array_sim; \
import src.sim.build_array_hi; \
import src.sim.build_channel_hi; \
import src.transfer.train_transfer; \
import src.experiments.run_groups; \
import src.train.pretrain; \
print('✅ 所有关键模块导入成功')" \
 && python -m pytest tests/ -q --tb=no -x

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["verify"]
