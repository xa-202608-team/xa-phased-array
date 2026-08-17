#!/usr/bin/env bash
# download_nasa_electronics.sh
#
# NASA PCoE 电子器件退化数据集断点续传下载脚本
#
# 支持：
#   - MOSFET Thermal Overstress Aging
#   - IGBT Accelerated Aging
#   - Capacitor Electrical Stress
#   - aria2c / curl / wget 自动选择
#   - 断点续传、重试、ZIP 完整性校验、SHA256 清单
#   - 可选自动解压（就地解压到组件目录下，zip 内顶层目录自然展开，与 zip 同级）
#
# 本地布局（与 nasa_electronics_manifest.yaml 对齐）：
#   data/raw/phased_array/
#     ├── MOSFET Thermal Overstress Aging/13_MOSFET_Thermal_Overstress_Aging.zip
#     ├── IGBT Acclerated Aging/IGBTAgingData_04022009.zip
#     └── Capacitor Electrical Stress/12.+Capacitor+Electrical+Stress.zip
#   组件子目录名沿用下载产物原貌（含 "Acclerated" 拼写、URL 编码的 "+"），
#   以保持与已记录 SHA256 的逐字节对应，不要随手改名。
#
# 用法：
#   bash download_nasa_electronics.sh mosfet
#   bash download_nasa_electronics.sh igbt
#   bash download_nasa_electronics.sh capacitor
#   bash download_nasa_electronics.sh all
#
# 可选环境变量：
#   DATA_DIR=data/raw/phased_array     保存根目录（每个数据集落到其组件子目录下）
#   EXTRACT=1                         下载后就地解压到 <组件目录>/（zip 内顶层目录自然展开）
#   FORCE=1                           即使现有 ZIP 校验通过也重新下载
#   CONNECTIONS=8                     aria2c 并发连接数
#
# 示例：
#   DATA_DIR="$PWD/data/raw/phased_array" EXTRACT=1 \
#     bash download_nasa_electronics.sh all

set -Eeuo pipefail

DATA_DIR="${DATA_DIR:-data/raw/phased_array}"
EXTRACT="${EXTRACT:-0}"
FORCE="${FORCE:-0}"
CONNECTIONS="${CONNECTIONS:-8}"

NASA_REPOSITORY_URL="https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/"

declare -A URLS=(
  [mosfet]="https://phm-datasets.s3.amazonaws.com/NASA/13.+MOSFET+Thermal+Overstress+Aging.zip"
  [igbt]="https://phm-datasets.s3.amazonaws.com/NASA/8.+IGBT+Accelerated+Aging.zip"
  [capacitor]="https://phm-datasets.s3.amazonaws.com/NASA/12.+Capacitor+Electrical+Stress.zip"
)

# 本地保存文件名。MOSFET 用规范化下划线名；IGBT/Capacitor 沿用实际下载产物名
# （IGBTAgingData_04022009.zip = 包内顶层目录名；Capacitor 保留 NASA S3 的 URL 编码 "+"）。
declare -A FILENAMES=(
  [mosfet]="13_MOSFET_Thermal_Overstress_Aging.zip"
  [igbt]="IGBTAgingData_04022009.zip"
  [capacitor]="12.+Capacitor+Electrical+Stress.zip"
)

# 每个数据集在 DATA_DIR 下的组件子目录（与磁盘现状逐字一致）。
declare -A SUBDIRS=(
  [mosfet]="MOSFET Thermal Overstress Aging"
  [igbt]="IGBT Acclerated Aging"
  [capacitor]="Capacitor Electrical Stress"
)

declare -A TITLES=(
  [mosfet]="NASA MOSFET Thermal Overstress Aging"
  [igbt]="NASA IGBT Accelerated Aging"
  [capacitor]="NASA Capacitor Electrical Stress"
)

usage() {
  cat <<'EOF'
NASA PCoE 电子器件退化数据集下载器

用法：
  bash download_nasa_electronics.sh <mosfet|igbt|capacitor|all|list>

命令：
  mosfet      下载 MOSFET Thermal Overstress Aging
  igbt        下载 IGBT Accelerated Aging
  capacitor   下载 Capacitor Electrical Stress
  all         下载以上全部数据集
  list        显示数据集名称和官方地址

环境变量：
  DATA_DIR    保存根目录，默认 data/raw/phased_array
              每个数据集落到其组件子目录下（如 data/raw/phased_array/MOSFET Thermal Overstress Aging/）
  EXTRACT=1   下载后就地解压到 <组件目录>/（与 zip 同级）
  FORCE=1     强制重新下载
  CONNECTIONS aria2c 并发连接数，默认 8

示例：
  bash download_nasa_electronics.sh mosfet
  DATA_DIR=/mnt/data/phased_array EXTRACT=1 bash download_nasa_electronics.sh all
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

verify_zip() {
  local file="$1"

  [[ -s "$file" ]] || return 1

  if have unzip; then
    unzip -tq "$file" >/dev/null 2>&1
    return $?
  fi

  if have python3; then
    python3 - "$file" <<'PY'
import sys
import zipfile

path = sys.argv[1]
try:
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad is not None:
            print(f"损坏成员: {bad}", file=sys.stderr)
            raise SystemExit(1)
except (OSError, zipfile.BadZipFile) as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)
PY
    return $?
  fi

  die "无法校验 ZIP：请安装 unzip 或 python3。"
}

download_with_aria2() {
  local url="$1"
  local output="$2"
  local dir name
  dir="$(dirname "$output")"
  name="$(basename "$output")"

  aria2c \
    --continue=true \
    --max-connection-per-server="$CONNECTIONS" \
    --split="$CONNECTIONS" \
    --min-split-size=4M \
    --max-tries=20 \
    --retry-wait=5 \
    --timeout=60 \
    --connect-timeout=30 \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --file-allocation=none \
    --dir="$dir" \
    --out="$name" \
    "$url"
}

download_with_curl() {
  local url="$1"
  local output="$2"

  # -C -：从已有文件末尾断点续传
  # --retry-all-errors：网络抖动、部分 4xx/5xx 时重试
  curl \
    --location \
    --fail \
    --continue-at - \
    --retry 20 \
    --retry-delay 5 \
    --retry-all-errors \
    --connect-timeout 30 \
    --speed-time 60 \
    --speed-limit 1024 \
    --output "$output" \
    "$url"
}

download_with_wget() {
  local url="$1"
  local output="$2"

  wget \
    --continue \
    --tries=20 \
    --timeout=60 \
    --waitretry=5 \
    --output-document="$output" \
    "$url"
}

download_file() {
  local url="$1"
  local output="$2"

  if have aria2c; then
    log "使用 aria2c 断点续传。"
    download_with_aria2 "$url" "$output"
  elif have curl; then
    log "使用 curl 断点续传。"
    download_with_curl "$url" "$output"
  elif have wget; then
    log "使用 wget 断点续传。"
    download_with_wget "$url" "$output"
  else
    die "未找到 aria2c、curl 或 wget。请至少安装一个下载工具。"
  fi
}

write_hash() {
  local file="$1"
  local hash_file="${file}.sha256"

  if have sha256sum; then
    (
      cd "$(dirname "$file")"
      sha256sum "$(basename "$file")" > "$(basename "$hash_file")"
    )
  elif have shasum; then
    (
      cd "$(dirname "$file")"
      shasum -a 256 "$(basename "$file")" > "$(basename "$hash_file")"
    )
  elif have python3; then
    python3 - "$file" "$hash_file" <<'PY'
import hashlib
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])

h = hashlib.sha256()
with src.open("rb") as f:
    for block in iter(lambda: f.read(1024 * 1024), b""):
        h.update(block)

dst.write_text(f"{h.hexdigest()}  {src.name}\n", encoding="utf-8")
PY
  else
    log "警告：没有 sha256sum、shasum 或 python3，跳过 SHA256。"
    return
  fi

  log "SHA256 已写入：$hash_file"
}

extract_zip() {
  local file="$1"
  local key="$2"
  local subdir="${SUBDIRS[$key]}"
  local out_dir="$DATA_DIR/$subdir"

  mkdir -p "$out_dir"

  if have unzip; then
    unzip -q -o "$file" -d "$out_dir"
  elif have python3; then
    python3 - "$file" "$out_dir" <<'PY'
import pathlib
import sys
import zipfile

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(src) as zf:
    zf.extractall(dst)
PY
  else
    die "无法解压：请安装 unzip 或 python3。"
  fi

  log "已解压到：$out_dir"
}

download_dataset() {
  local key="$1"
  local url="${URLS[$key]}"
  local filename="${FILENAMES[$key]}"
  local subdir="${SUBDIRS[$key]}"
  local title="${TITLES[$key]}"
  local dest_dir="$DATA_DIR/$subdir"
  local output="$dest_dir/$filename"

  mkdir -p "$dest_dir"

  printf '\n'
  log "数据集：$title"
  log "官方地址：$url"
  log "保存路径：$output"

  if [[ -f "$output" && "$FORCE" != "1" ]]; then
    log "检测到已有文件，先执行完整性校验。"
    if verify_zip "$output"; then
      log "已有 ZIP 完整，跳过下载。设置 FORCE=1 可强制重下。"
      write_hash "$output"
      if [[ "$EXTRACT" == "1" ]]; then
        extract_zip "$output" "$key"
      fi
      return 0
    fi
    log "已有文件不完整，将从断点继续下载。"
  elif [[ "$FORCE" == "1" && -f "$output" ]]; then
    log "FORCE=1：删除现有文件并重新下载。"
    rm -f "$output" "${output}.aria2"
  fi

  if ! download_file "$url" "$output"; then
    printf '\n下载失败：%s\n' "$title" >&2
    printf 'NASA 数据仓库入口：%s\n' "$NASA_REPOSITORY_URL" >&2
    if [[ "$key" == "igbt" ]]; then
      cat >&2 <<'EOF'

IGBT 可选 Kaggle 镜像（需要 Kaggle API 凭据，仅作传输通道，不可作为真源）：
  pip install kaggle
  kaggle datasets download \
    -d vignesh9147/igbt-accelerated-aging-data-set \
    -p "data/raw/phased_array/IGBT Acclerated Aging/igbt_kaggle"
EOF
    fi
    return 1
  fi

  log "下载完成，开始校验 ZIP。"
  if ! verify_zip "$output"; then
    rm -f "$output"
    die "ZIP 校验失败，已删除损坏文件。请重新运行脚本。"
  fi

  log "ZIP 完整性校验通过。"
  write_hash "$output"

  if [[ "$EXTRACT" == "1" ]]; then
    extract_zip "$output" "$key"
  fi
}

list_datasets() {
  printf '%-12s  %s\n' "KEY" "OFFICIAL URL"
  printf '%-12s  %s\n' "-----------" "------------"
  for key in mosfet igbt capacitor; do
    printf '%-12s  %s\n' "$key" "${URLS[$key]}"
  done
}

main() {
  local target="${1:-}"

  case "$target" in
    mosfet|igbt|capacitor)
      download_dataset "$target"
      ;;
    all)
      local failures=0
      for key in mosfet igbt capacitor; do
        if ! download_dataset "$key"; then
          failures=$((failures + 1))
        fi
      done
      if (( failures > 0 )); then
        die "$failures 个数据集下载失败，请查看上方日志后重试。"
      fi
      ;;
    list)
      list_datasets
      ;;
    -h|--help|help|"")
      usage
      ;;
    *)
      usage
      die "未知命令：$target"
      ;;
  esac
}

main "$@"
