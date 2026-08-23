# =====================================================================
# 相控阵组件 - 复现镜像 (Python 3.12 + CUDA torch)
#   构建时验证: 导入检查 + 契约 Schema 快照 + 完整 pytest (依赖缺失测试自带 skip)
#   运行时验证: docker run --rm xa-phased-array:v0.3.0-rc.1 verify
#
# 工件边界 (契约 v1.1):
#   - canonical H5 与 .pt 不烘焙进镜像、不入 Git
#   - /artifacts/data 与 /artifacts/checkpoints 只读挂载 (缺失时走 synthetic 源域,
#     输出 manifest 标 source_mode=synthetic)
#   - /outputs 唯一可写输出目录
#   - verify 只跑代码/fixture/Schema/导入检查, 不依赖大数据工件
#
# GPU: 宿主机需 NVIDIA Container Toolkit; 无 GPU 自动 fallback CPU
# =====================================================================
FROM python:3.12-slim

# torch 变体: 默认 CUDA (cu130); 无 GPU / NVIDIA 源不可达环境可构建 CPU 验证镜像:
#   docker build --build-arg TORCH_EXTRA_INDEX=https://download.pytorch.org/whl/cpu -t ...
ARG TORCH_EXTRA_INDEX=https://download.pytorch.org/whl/cu130
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) 先装依赖 (torch 按 TORCH_EXTRA_INDEX 选 CPU/CUDA 变体)
#    torch 的纯 Python 依赖先从主镜像源装好, 避免 pip 经 extra-index 解析时
#    落到 PyPI 官方域名 (容器内常不可达); torch wheel 本体从 pytorch 官方源取
COPY requirements.txt /app/
RUN pip install --no-cache-dir --default-timeout=300 --retries=5 \
      filelock typing-extensions sympy networkx jinja2 fsspec markupsafe \
      --index-url "$PIP_INDEX" --trusted-host pypi.tuna.tsinghua.edu.cn \
 && pip install --no-cache-dir --default-timeout=300 --retries=5 torch==2.11.0 \
      --index-url "$PIP_INDEX" \
      --extra-index-url "$TORCH_EXTRA_INDEX" \
      --trusted-host pypi.tuna.tsinghua.edu.cn \
      --trusted-host download.pytorch.org \
 && pip install --no-cache-dir --default-timeout=300 --retries=5 -r requirements.txt \
      --index-url "$PIP_INDEX" \
      --trusted-host pypi.tuna.tsinghua.edu.cn

# 2) 非 root 用户
RUN useradd -m -u 1000 appuser

# 3) 拷贝代码 (不拷贝 data/ 与 checkpoints/ — 大型工件不进镜像层)
#    data/simulated 与 features 由 entrypoint 运行时再生 (仿真器自包含)
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser configs/ /app/configs/
COPY --chown=appuser:appuser tests/ /app/tests/
COPY --chown=appuser:appuser schemas/ /app/schemas/
COPY --chown=appuser:appuser component/ /app/component/
COPY --chown=appuser:appuser docs/ /app/docs/
COPY --chown=appuser:appuser scripts/ /app/scripts/
COPY --chown=appuser:appuser pytest.ini /app/
COPY --chown=appuser:appuser scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh \
 && mkdir -p /artifacts/data /artifacts/checkpoints /outputs /app/data /app/checkpoints \
 && chown -R appuser:appuser /artifacts /outputs /app/data /app/checkpoints

USER appuser

ENV PYTHONPATH=/app

# 构建时烘焙源码 commit (reproduce_judge/full 的 manifest git_commit 用;
# 容器内无 .git, 不烘焙则 judge 拒绝伪造而失败——构建命令示例:
#   docker build --build-arg XA_GIT_COMMIT=$(git rev-parse HEAD) -t xa-phased-array:... .
# 放在依赖层之后, 改变 commit 不会击穿 pip 层缓存)
ARG XA_GIT_COMMIT=
ENV XA_GIT_COMMIT=${XA_GIT_COMMIT}

# 4) 构建时验证: 契约 Schema 快照 + 导入检查 + pytest (缺失外部数据的测试自带 skip)
RUN python -c "\
import json, glob, jsonschema; \
schemas = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob('schemas/*.schema.json'))]; \
assert len(schemas) == 9, f'expect 9 schemas (8 contract snapshots + rul-prediction), got {len(schemas)}'; \
[jsonschema.validators.validator_for(s) for s in schemas]; \
print('contract schemas ok:', len(schemas))" \
 && python -c "\
import src.sim.phased_array_sim; \
import src.sim.build_array_hi; \
import src.sim.build_channel_hi; \
import src.transfer.train_transfer; \
import src.experiments.run_groups; \
import src.train.pretrain; \
import component.predictor; \
print('all key modules import ok')" \
 && python -m pytest tests/ -q --tb=no -p no:cacheprovider

# 工件挂载点: /artifacts/* 由调用方只读挂载, /outputs 为唯一可写输出
VOLUME ["/outputs"]

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["verify"]
