#!/usr/bin/env bash
# 一键辨识脚本：CSV -> NPZ -> MuJoCo 粗拟合 -> Isaac 精修 -> 验证
# 用法：
#   ./scripts/run_identification.sh <RUN_ID> [选项]
#
# 选项：
#   --fast           MuJoCo 用快速配置(1400候选/关节)，Isaac 降到 24576候选/关节
#   --isaac-only     跳过 CSV/NPZ、MuJoCo 粗拟合和 MuJoCo 验证，只跑 Isaac 精修与验证
#   --skip-isaac     只跑 MuJoCo 粗拟合和验证，跳过 Isaac 精修
#   --mujoco-only    等同于 --skip-isaac
#   --device DEVICE  Isaac 使用的物理 GPU，默认 cuda:0（多卡服务器用 cuda:1 等）
#   --num-envs N     Isaac 每个 chunk 的并行 env 数，默认 6144；显存不够可降到 4096/3072
#   --start-joint N  Isaac 从第 N 个关节继续；前面的关节从 results/isaac_trace 恢复
#   --jobs N         MuJoCo 并行进程数，默认 6
#   --mujoco-samples N                覆盖 MuJoCo 每个离散组合的随机样本数(默认普通=500, fast=200)
#   --isaac-candidates N              覆盖 Isaac 每关节总候选数(默认普通=49152, fast=24576)
#   --refinement-rounds N             覆盖 Isaac 精修轮数(默认 2)
#   --refine-candidates-per-round N   固定每轮候选数，此时 --isaac-candidates 只作为日志参考
#
# 路径可通过环境变量覆盖（不改脚本适配不同机器）：
#   ISAACLAB   经确认兼容的 isaaclab.sh 路径，或使用 ISAACLAB_PATH
#   ISAAC_USD  机器人 USD 路径，默认使用本仓库内置 my_robot.usd
#   PYTHON     python 可执行文件，默认 python3
#
# 示例：
#   ./scripts/run_identification.sh 20260611_201935
#   ./scripts/run_identification.sh 20260611_201935 --isaac-only
#   ./scripts/run_identification.sh 20260611_201935 --fast --device cuda:1
#   ISAACLAB=/path/to/IsaacLab/isaaclab.sh ISAAC_USD=/path/to/my_robot.usd \
#     ./scripts/run_identification.sh 20260611_201935

set -euo pipefail

# ──────────────────────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "用法: $0 <RUN_ID> [--fast] [--isaac-only] [--skip-isaac] [--mujoco-only] [--device cuda:N] [--num-envs N] [--start-joint N] [--jobs N] [--mujoco-samples N] [--isaac-candidates N] [--refinement-rounds N] [--refine-candidates-per-round N]"
    exit 1
fi

RUN_ID="$1"
shift

FAST=0
ISAAC_ONLY=0
SKIP_ISAAC=0
ISAAC_DEVICE="cuda:0"
ISAAC_DEVICE_EXPLICIT=0
ISAAC_NUM_ENVS=6144
ISAAC_START_JOINT=1
JOBS=6
MUJOCO_RANDOM_SAMPLES=""
ISAAC_CANDIDATES_OVERRIDE=""
ISAAC_REFINEMENT_ROUNDS=""
ISAAC_REFINE_CANDIDATES_PER_ROUND=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast)        FAST=1 ;;
        --isaac-only)  ISAAC_ONLY=1 ;;
        --skip-isaac)  SKIP_ISAAC=1 ;;
        --mujoco-only) SKIP_ISAAC=1 ;;
        --device)      ISAAC_DEVICE="$2"; ISAAC_DEVICE_EXPLICIT=1; shift ;;
        --num-envs)    ISAAC_NUM_ENVS="$2"; shift ;;
        --start-joint) ISAAC_START_JOINT="$2"; shift ;;
        --jobs)        JOBS="$2"; shift ;;
        --mujoco-samples) MUJOCO_RANDOM_SAMPLES="$2"; shift ;;
        --isaac-candidates) ISAAC_CANDIDATES_OVERRIDE="$2"; shift ;;
        --refinement-rounds) ISAAC_REFINEMENT_ROUNDS="$2"; shift ;;
        --refine-candidates-per-round) ISAAC_REFINE_CANDIDATES_PER_ROUND="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

# ──────────────────────────────────────────────────────────────
# 路径配置（可通过同名环境变量覆盖）
# ──────────────────────────────────────────────────────────────
ID_REPO="$(cd "$(dirname "$0")/.." && pwd)"
# An explicit ID_CSV_DIR is used as given; only the default searches both data roots.
CSV_DIR="$ID_REPO/identification_data/$RUN_ID/csv"
if [[ -n "${ID_CSV_DIR:-}" ]]; then
    CSV_DIR="$ID_CSV_DIR"
elif [[ ! -d "$CSV_DIR" && -d "$ID_REPO/arx_id_data/$RUN_ID/csv" ]]; then
    CSV_DIR="$ID_REPO/arx_id_data/$RUN_ID/csv"
fi
RUN_OUTPUT="${ID_OUTPUT_DIR:-$ID_REPO/outputs/pd/$RUN_ID}"
NPZ_DIR="$RUN_OUTPUT/converted"
STEP_NPZ_DIR="$RUN_OUTPUT/heldout_step_converted"
OUT_DIR="$RUN_OUTPUT/results"

EMA_ALPHA=0.05

if [[ $FAST -eq 1 ]]; then
    ID_CONFIG="$ID_REPO/config/arx_r5_smooth_pd_identify_fast.yaml"
    ISAAC_CANDIDATES=24576
    DEFAULT_MUJOCO_RANDOM_SAMPLES=200
else
    ID_CONFIG="$ID_REPO/config/arx_r5_smooth_pd_identify.yaml"
    ISAAC_CANDIDATES=49152
    DEFAULT_MUJOCO_RANDOM_SAMPLES=500
fi
if [[ -z "$MUJOCO_RANDOM_SAMPLES" ]]; then
    MUJOCO_RANDOM_SAMPLES="$DEFAULT_MUJOCO_RANDOM_SAMPLES"
fi
if [[ -n "$ISAAC_CANDIDATES_OVERRIDE" ]]; then
    ISAAC_CANDIDATES="$ISAAC_CANDIDATES_OVERRIDE"
fi
if [[ -z "$ISAAC_REFINEMENT_ROUNDS" ]]; then
    ISAAC_REFINEMENT_ROUNDS=2
fi

B2ARX_CONFIG="$ID_REPO/models/b2arx_mj/b2arx_mj.yaml"
DEFAULT_ISAAC_USD="$ID_REPO/arx_actuator_identification/my_robot/my_b2arx/my_robot.usd"
ISAAC_USD="${ISAAC_USD:-$DEFAULT_ISAAC_USD}"
ISAACLAB="${ISAACLAB:-${ISAACLAB_PATH:+$ISAACLAB_PATH/isaaclab.sh}}"
PYTHON="${PYTHON:-python3}"

ISAAC_RUNTIME_DEVICE="$ISAAC_DEVICE"
ISAAC_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ISAAC_DEVICE_DESC="$ISAAC_DEVICE"
if [[ "$ISAAC_DEVICE" =~ ^cuda:([0-9]+)$ ]]; then
    if [[ $ISAAC_DEVICE_EXPLICIT -eq 1 ]]; then
        ISAAC_CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
        ISAAC_RUNTIME_DEVICE="cuda:0"
        ISAAC_DEVICE_DESC="physical $ISAAC_DEVICE -> CUDA_VISIBLE_DEVICES=$ISAAC_CUDA_VISIBLE_DEVICES, runtime cuda:0"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        ISAAC_RUNTIME_DEVICE="cuda:0"
        ISAAC_DEVICE_DESC="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, runtime cuda:0"
    else
        ISAAC_CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
        ISAAC_RUNTIME_DEVICE="cuda:0"
        ISAAC_DEVICE_DESC="physical $ISAAC_DEVICE -> CUDA_VISIBLE_DEVICES=$ISAAC_CUDA_VISIBLE_DEVICES, runtime cuda:0"
    fi
elif [[ "$ISAAC_DEVICE" =~ ^[0-9]+$ ]]; then
    ISAAC_CUDA_VISIBLE_DEVICES="$ISAAC_DEVICE"
    ISAAC_RUNTIME_DEVICE="cuda:0"
    ISAAC_DEVICE_DESC="physical cuda:$ISAAC_DEVICE -> CUDA_VISIBLE_DEVICES=$ISAAC_CUDA_VISIBLE_DEVICES, runtime cuda:0"
fi

# ──────────────────────────────────────────────────────────────
# 输出路径
# ──────────────────────────────────────────────────────────────
MUJOCO_PARAMS="$OUT_DIR/actuator_params_mujoco.yaml"
MUJOCO_REPORT="$OUT_DIR/actuator_params_mujoco_fit_report.yaml"
MUJOCO_TRACE="$OUT_DIR/mujoco_trace"
MUJOCO_HELDOUT="$OUT_DIR/mujoco_heldout_step"

ISAAC_PARAMS="$OUT_DIR/actuator_params_isaac.yaml"
ISAAC_REPORT="$OUT_DIR/actuator_params_isaac_fit_report.yaml"
ISAAC_TRACE="$OUT_DIR/isaac_trace"
ISAAC_HELDOUT="$OUT_DIR/isaac_heldout_step"

# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────
log_stage() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════"
}

# ──────────────────────────────────────────────────────────────
# 前置检查
# ──────────────────────────────────────────────────────────────
log_stage "前置检查  RUN_ID=$RUN_ID  FAST=$FAST  ISAAC_ONLY=$ISAAC_ONLY  SKIP_ISAAC=$SKIP_ISAAC"

if [[ $ISAAC_ONLY -eq 1 && $SKIP_ISAAC -eq 1 ]]; then
    echo "错误: --isaac-only 与 --skip-isaac/--mujoco-only 不能同时使用"
    exit 1
fi
if ! [[ "$ISAAC_START_JOINT" =~ ^[1-6]$ ]]; then
    echo "错误: --start-joint 必须是 1..6，当前为: $ISAAC_START_JOINT"
    exit 1
fi

if [[ $ISAAC_ONLY -eq 0 ]]; then
    if [[ ! -d "$CSV_DIR" ]]; then
        echo "错误: CSV 目录不存在: $CSV_DIR"
        exit 1
    fi

    # 检查 6 个关节 chirp 数据
    for j in 0 1 2 3 4 5; do
        if [[ ! -f "$CSV_DIR/joint${j}_chirp.csv" ]]; then
            echo "错误: 缺少 joint${j}_chirp.csv"
            exit 1
        fi
    done
else
    if [[ ! -d "$NPZ_DIR/smooth" ]]; then
        echo "错误: Isaac-only 需要已有 smooth NPZ 目录: $NPZ_DIR/smooth"
        exit 1
    fi
    if [[ ! -f "$MUJOCO_REPORT" ]]; then
        echo "错误: Isaac-only 需要已有 MuJoCo fit report: $MUJOCO_REPORT"
        exit 1
    fi
    if [[ ! -d "$MUJOCO_TRACE" ]]; then
        echo "错误: Isaac-only 需要已有 MuJoCo trace 目录: $MUJOCO_TRACE"
        exit 1
    fi
fi

if [[ "$ISAAC_START_JOINT" != "1" ]]; then
    if [[ ! -d "$ISAAC_TRACE" ]]; then
        echo "错误: --start-joint $ISAAC_START_JOINT 需要已有 Isaac trace 目录: $ISAAC_TRACE"
        exit 1
    fi
    for ((j=1; j<ISAAC_START_JOINT; j++)); do
        if [[ ! -f "$ISAAC_TRACE/joint_${j}_isaac_top_trace.csv" ]]; then
            echo "错误: --start-joint $ISAAC_START_JOINT 缺少已完成关节 trace: $ISAAC_TRACE/joint_${j}_isaac_top_trace.csv"
            exit 1
        fi
    done
fi

if [[ $SKIP_ISAAC -eq 0 ]]; then
    if [[ ! -f "$ISAAC_USD" ]]; then
        echo "错误: Isaac USD 不存在: $ISAAC_USD"
        echo "提示: 如只跑 MuJoCo 可加 --skip-isaac"
        exit 1
    fi
    if [[ ! -x "$ISAACLAB" ]]; then
        echo "错误: 请通过 ISAACLAB 指定兼容的 isaaclab.sh。旧 PhysX 后端版本说明见 docs/versions.md: $ISAACLAB"
        exit 1
    fi
fi

mkdir -p "$NPZ_DIR" "$STEP_NPZ_DIR" "$OUT_DIR"

if [[ $ISAAC_ONLY -eq 1 ]]; then
    log_stage "跳过步骤 1-3  直接 Isaac 精修"
else

# ──────────────────────────────────────────────────────────────
# 步骤 1: CSV -> NPZ
# ──────────────────────────────────────────────────────────────
log_stage "步骤 1/5  CSV -> NPZ"

cd "$ID_REPO"

$PYTHON -m arx_actuator_identification.cli.csv_to_npz \
    "$CSV_DIR"/joint*_chirp.csv \
    --cmd-source both \
    --out-dir "$NPZ_DIR"

# step held-out（joint5 step 截断也照常转，验证时注意行数）
STEP_CSVS=("$CSV_DIR"/joint*_step.csv)
if [[ ${#STEP_CSVS[@]} -gt 0 && -f "${STEP_CSVS[0]}" ]]; then
    $PYTHON -m arx_actuator_identification.cli.csv_to_npz \
        "$CSV_DIR"/joint*_step.csv \
        --cmd-source both \
        --out-dir "$STEP_NPZ_DIR"
else
    echo "警告: 未找到 step CSV，跳过 held-out 转换"
fi

# ──────────────────────────────────────────────────────────────
# 步骤 2: MuJoCo 粗拟合
# ──────────────────────────────────────────────────────────────
MUJOCO_EXTRA_ARGS=()
if [[ -n "$MUJOCO_RANDOM_SAMPLES" ]]; then
    MUJOCO_EXTRA_ARGS+=(--random-samples "$MUJOCO_RANDOM_SAMPLES")
fi

log_stage "步骤 2/5  MuJoCo 粗拟合  config=$(basename $ID_CONFIG)  random_samples=${MUJOCO_RANDOM_SAMPLES:-config}"

$PYTHON -m arx_actuator_identification.cli.fit_all \
    --backend mujoco \
    --data-dir "$NPZ_DIR/smooth" \
    --config "$ID_CONFIG" \
    --b2arx-config "$B2ARX_CONFIG" \
    --output "$MUJOCO_PARAMS" \
    --report "$MUJOCO_REPORT" \
    --trace-dir "$MUJOCO_TRACE" \
    --cmd-source smooth \
    --ema-alpha "$EMA_ALPHA" \
    "${MUJOCO_EXTRA_ARGS[@]}" \
    --jobs "$JOBS"

# ──────────────────────────────────────────────────────────────
# 步骤 3: MuJoCo held-out 验证
# ──────────────────────────────────────────────────────────────
log_stage "步骤 3/5  MuJoCo held-out step 验证"

if [[ -d "$STEP_NPZ_DIR/smooth" ]]; then
    $PYTHON -m arx_actuator_identification.cli.validate_mujoco_fit \
        --data-dir "$STEP_NPZ_DIR/smooth" \
        --params "$MUJOCO_PARAMS" \
        --output-dir "$MUJOCO_HELDOUT" \
        --b2arx-config "$B2ARX_CONFIG"
    echo "MuJoCo 验证报告: $MUJOCO_HELDOUT/validation_report.yaml"
else
    echo "警告: 未找到 held-out NPZ，跳过 MuJoCo 验证"
fi

fi

# ──────────────────────────────────────────────────────────────
# 步骤 4: Isaac 精修
# ──────────────────────────────────────────────────────────────
if [[ $SKIP_ISAAC -eq 1 ]]; then
    log_stage "跳过 Isaac 精修（--skip-isaac）"
    echo "最终参数(MuJoCo): $MUJOCO_PARAMS"
    exit 0
fi

ISAAC_EXTRA_ARGS=()
if [[ -n "$ISAAC_REFINEMENT_ROUNDS" ]]; then
    ISAAC_EXTRA_ARGS+=(--refinement-rounds "$ISAAC_REFINEMENT_ROUNDS")
fi
if [[ -n "$ISAAC_REFINE_CANDIDATES_PER_ROUND" ]]; then
    ISAAC_EXTRA_ARGS+=(--refine-candidates-per-round "$ISAAC_REFINE_CANDIDATES_PER_ROUND")
fi
if [[ "$ISAAC_START_JOINT" != "1" ]]; then
    ISAAC_EXTRA_ARGS+=(--start-joint "$ISAAC_START_JOINT")
fi

log_stage "步骤 4/5  Isaac 精修  candidates-per-joint=$ISAAC_CANDIDATES  refinement_rounds=${ISAAC_REFINEMENT_ROUNDS:-config}  refine_candidates_per_round=${ISAAC_REFINE_CANDIDATES_PER_ROUND:-config}  num_envs=$ISAAC_NUM_ENVS  start_joint=$ISAAC_START_JOINT"
echo "Isaac device: $ISAAC_DEVICE_DESC"

rm -f "$ISAAC_PARAMS" "$ISAAC_REPORT"

env CUDA_VISIBLE_DEVICES="$ISAAC_CUDA_VISIBLE_DEVICES" "$ISAACLAB" -p -m arx_actuator_identification.cli.fit_all_isaac \
    --data-dir "$NPZ_DIR/smooth" \
    --mujoco-report "$MUJOCO_REPORT" \
    --mujoco-trace-dir "$MUJOCO_TRACE" \
    --config "$ID_CONFIG" \
    --usd-path "$ISAAC_USD" \
    --output "$ISAAC_PARAMS" \
    --report "$ISAAC_REPORT" \
    --trace-dir "$ISAAC_TRACE" \
    --num-envs "$ISAAC_NUM_ENVS" \
    --candidates-per-joint "$ISAAC_CANDIDATES" \
    "${ISAAC_EXTRA_ARGS[@]}" \
    --device "$ISAAC_RUNTIME_DEVICE" \
    --cmd-source smooth \
    --ema-alpha "$EMA_ALPHA" \
    --headless

if [[ ! -f "$ISAAC_PARAMS" || ! -f "$ISAAC_REPORT" ]]; then
    echo "错误: Isaac 精修未生成结果，停止后续验证。"
    echo "  缺失检查: $ISAAC_PARAMS"
    echo "  缺失检查: $ISAAC_REPORT"
    echo "提示: 如果仍然 CUDA OOM，可重跑并降低 --num-envs，例如 --num-envs 4096 或 --num-envs 3072。"
    exit 1
fi

# ──────────────────────────────────────────────────────────────
# 步骤 5: Isaac held-out 验证
# ──────────────────────────────────────────────────────────────
log_stage "步骤 5/5  Isaac held-out step 验证"

if [[ -d "$STEP_NPZ_DIR/smooth" ]]; then
    env CUDA_VISIBLE_DEVICES="$ISAAC_CUDA_VISIBLE_DEVICES" "$ISAACLAB" -p -m arx_actuator_identification.cli.validate_isaac_fit \
        --data-dir "$STEP_NPZ_DIR/smooth" \
        --params "$ISAAC_PARAMS" \
        --fit-report "$ISAAC_REPORT" \
        --trace-dir "$ISAAC_TRACE" \
        --output-dir "$ISAAC_HELDOUT" \
        --usd-path "$ISAAC_USD" \
        --device "$ISAAC_RUNTIME_DEVICE" \
        --headless
    echo "Isaac 验证报告: $ISAAC_HELDOUT/validation_report.yaml"
else
    echo "警告: 未找到 held-out NPZ，跳过 Isaac 验证"
fi

# ──────────────────────────────────────────────────────────────
# 完成
# ──────────────────────────────────────────────────────────────
log_stage "完成"
echo "最终参数(Isaac): $ISAAC_PARAMS"
echo "MuJoCo 验证:     $MUJOCO_HELDOUT/validation_report.yaml"
echo "Isaac  验证:     $ISAAC_HELDOUT/validation_report.yaml"
