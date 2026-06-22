from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
VENV_ACTIVATE = ROOT / ".venv_tf310_dml" / "Scripts" / "Activate.ps1"


@dataclass
class TaskState:
  running: bool = False
  returncode: int | None = None
  log: list[str] = field(default_factory=list)
  queue: Queue[str] | None = None
  process: subprocess.Popen[str] | None = None
  thread: threading.Thread | None = None


if "task" not in st.session_state:
  st.session_state.task = TaskState()


GUI_TASKS: dict[str, dict[str, Any]] = {
    "prepare": {
        "label": "数据预处理",
        "script": ROOT / "scripts" / "prepare_cicids2017_train_split.py",
        "args": [
            {"key": "--csv-dir", "label": "CSV 目录", "type": "text", "default": ""},
            {"key": "--output-dir", "label": "输出目录", "type": "text", "default": "data/processed"},
            {"key": "--sample-per-file", "label": "每文件抽样", "type": "number", "default": 20000},
            {"key": "--train-fraction", "label": "训练集比例", "type": "number", "default": 0.8, "step": 0.05},
            {"key": "--max-train-samples", "label": "训练集最大样本数", "type": "number", "default": 0},
            {"key": "--min-per-class", "label": "每类最少样本", "type": "number", "default": 10},
            {"key": "--random-state", "label": "随机种子", "type": "number", "default": 42},
            {"key": "--demo", "label": "使用 demo 数据", "type": "checkbox", "default": False},
        ],
    },
    "pcap_to_csv": {
        "label": "PCAP → 流特征 CSV",
        "script": ROOT / "scripts" / "pcaptoflowcsv.py",
        "args": [
            {"key": "--pcap", "label": "PCAP 文件", "type": "text", "default": ""},
            {"key": "--output", "label": "输出 CSV", "type": "text", "default": "flows.csv"},
            {"key": "--label", "label": "标签", "type": "text", "default": "BENIGN"},
            {"key": "--metadata", "label": "metadata", "type": "text", "default": "data/processed/metadata.json"},
            {"key": "--limit", "label": "限制条数", "type": "number", "default": 0},
        ],
    },
    "train_cnn": {
        "label": "CNN 联邦训练",
        "script": ROOT / "federated" / "train_fl.py",
        "args": [
            {"key": "--data-dir", "label": "数据目录", "type": "text", "default": "data/processed"},
            {"key": "--artifacts-dir", "label": "模型输出目录", "type": "text", "default": "artifacts"},
            {"key": "--rounds", "label": "训练轮次", "type": "number", "default": 15},
            {"key": "--batch-size", "label": "批大小", "type": "number", "default": 256},
            {"key": "--client-epochs", "label": "客户端 epoch", "type": "number", "default": 1},
            {"key": "--client-lr", "label": "客户端学习率", "type": "number", "default": 0.02, "step": 0.001},
            {"key": "--server-lr", "label": "服务器学习率", "type": "number", "default": 1.0, "step": 0.1},
            {"key": "--dropout", "label": "Dropout", "type": "number", "default": 0.0, "step": 0.05},
            {"key": "--hidden", "label": "隐藏层", "type": "text", "default": "256,128"},
            {"key": "--client-opt", "label": "客户端优化器", "type": "select", "default": "sgd", "options": ["sgd", "adam"]},
            {"key": "--loss", "label": "损失函数", "type": "select", "default": "crossentropy", "options": ["crossentropy", "focal"]},
            {"key": "--focal-gamma", "label": "Focal Gamma", "type": "number", "default": 2.0, "step": 0.1},
            {"key": "--round-lr-decay", "label": "轮次衰减", "type": "number", "default": 1.0, "step": 0.01},
            {"key": "--clipnorm", "label": "梯度裁剪", "type": "number", "default": 0.0, "step": 0.1},
            {"key": "--no-class-weight", "label": "关闭 class_weight", "type": "checkbox", "default": False},
            {"key": "--class-weight-power", "label": "class_weight 幂次", "type": "number", "default": 1.0, "step": 0.1},
        ],
    },
    "eval_cnn": {
        "label": "CNN 模型评估",
        "script": ROOT / "federated" / "inference.py",
        "args": [
            {"key": "--artifacts-dir", "label": "模型目录", "type": "text", "default": "artifacts"},
            {"key": "--processed-dir", "label": "数据目录", "type": "text", "default": "data/processed"},
            {"key": "--eval-npz", "label": "评估 NPZ", "type": "text", "default": "data/processed/test.npz"},
            {"key": "--eval-limit", "label": "评估样本数", "type": "number", "default": 5000},
            {"key": "--brief", "label": "简短输出", "type": "checkbox", "default": False},
            {"key": "--confusion", "label": "打印混淆矩阵", "type": "checkbox", "default": False},
            {"key": "--anomaly-threshold", "label": "异常阈值", "type": "number", "default": 0.5, "step": 0.05},
            {"key": "--eval-diagnostics", "label": "输出诊断", "type": "checkbox", "default": False},
        ],
    },
    "infer_csv": {
        "label": "真实 CSV 推理",
        "script": ROOT / "federated" / "inference_real_csv.py",
        "args": [
            {"key": "--csv", "label": "CSV 文件", "type": "text", "default": ""},
            {"key": "--artifacts-dir", "label": "模型目录", "type": "text", "default": "artifacts"},
            {"key": "--processed-dir", "label": "数据目录", "type": "text", "default": "data/processed"},
            {"key": "--label-col", "label": "标签列", "type": "text", "default": "Label"},
            {"key": "--topk", "label": "Top-K", "type": "number", "default": 1},
            {"key": "--limit", "label": "预测条数", "type": "number", "default": 0},
            {"key": "--show-prob", "label": "显示概率", "type": "checkbox", "default": False},
        ],
    },
}


def launch_command(command: list[str], cwd: Path, ignore_warnings: bool = False) -> None:
  state: TaskState = st.session_state.task
  state.running = True
  state.returncode = None
  state.log = []
  q: Queue[str] = Queue()
  state.queue = q

  env = os.environ.copy()
  env.setdefault("PYTHONUTF8", "1")
  env["VIRTUAL_ENV"] = str(ROOT / ".venv_tf310_dml")
  env["PATH"] = str(ROOT / ".venv_tf310_dml" / "Scripts") + os.pathsep + env.get("PATH", "")
  if ignore_warnings:
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    env["PYTHONWARNINGS"] = "ignore"

  def runner() -> None:
    try:
      proc = subprocess.Popen(
          command,
          cwd=str(cwd),
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
          encoding="utf-8",
          errors="replace",
          bufsize=1,
          env=env,
          shell=False,
      )
      state.process = proc
      if proc.stdout is None:
        q.put("[process-error] 无法读取子进程输出\n")
        return
      for line in proc.stdout:
        q.put(line)
      proc.wait()
      q.put(f"[process-exit] code={proc.returncode}\n")
    except Exception as exc:
      q.put(f"[process-error] {type(exc).__name__}: {exc}\n")
      q.put("[process-exit] code=1\n")

  thread = threading.Thread(target=runner, daemon=True)
  state.thread = thread
  thread.start()


def drain_logs() -> None:
  state: TaskState = st.session_state.task
  if state.queue is None:
    return
  try:
    while True:
      msg = state.queue.get_nowait()
      state.log.append(msg.rstrip("\n"))
      if msg.startswith("[process-exit]"):
        state.running = False
        if state.process is not None:
          state.returncode = state.process.returncode
  except Empty:
    pass


def build_command(task_id: str, params: dict[str, Any], python_exe: Path) -> list[str]:
  task = GUI_TASKS[task_id]
  command = [str(python_exe), str(task["script"])]
  for key, value in params.items():
    if value is True:
      command.append(key)
    elif value not in (None, False, ""):
      command.extend([key, str(value)])
  return command


def build_task_command(task_id: str, params: dict[str, Any], python_exe: Path, summary_json: Path) -> list[str]:
  command = build_command(task_id, params, python_exe)
  if task_id in {"eval_cnn", "infer_csv"}:
    command.extend(["--summary-json", str(summary_json)])
  return command


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
  if not path.is_file():
    return None
  try:
    with open(path, "r", encoding="utf-8") as f:
      return json.load(f)
  except Exception:
    return None


def show_summary_charts(task_id: str, summary: dict[str, Any]) -> None:
  if task_id == "eval_cnn":
    label_classes = summary.get("label_classes", [])
    true_counts = summary.get("true_class_counts", [])
    pred_counts = summary.get("pred_class_counts", [])
    if label_classes and (true_counts or pred_counts):
      df = pd.DataFrame({
          "类别": label_classes,
          "真实数量": true_counts,
          "预测数量": pred_counts,
      })
      st.subheader("类别分布")
      st.bar_chart(df.set_index("类别"))
    if "confusion_matrix" in summary and label_classes:
      st.subheader("混淆矩阵")
      cm = pd.DataFrame(summary.get("confusion_matrix", []), index=label_classes, columns=label_classes)
      st.dataframe(cm, use_container_width=True)
      st.bar_chart(cm)
    if "benign_count" in summary and "attack_count" in summary:
      st.subheader("异常 / 正常占比")
      pie_df = pd.DataFrame({
          "类别": ["正常", "异常"],
          "数量": [summary.get("benign_count", 0), summary.get("attack_count", 0)],
      })
      st.bar_chart(pie_df.set_index("类别"))
  elif task_id == "infer_csv":
    label_classes = summary.get("label_classes", [])
    pred_counts = summary.get("pred_class_counts", [])
    if label_classes and pred_counts:
      df = pd.DataFrame({"类别": label_classes, "预测数量": pred_counts})
      st.subheader("预测类别分布")
      st.bar_chart(df.set_index("类别"))
    st.subheader("置信度分布")
    stats = pd.DataFrame({
        "指标": ["平均Top-1概率", "最小Top-1概率", "最大Top-1概率"],
        "数值": [summary.get("avg_top1_prob", 0.0), summary.get("min_top1_prob", 0.0), summary.get("max_top1_prob", 0.0)],
    })
    st.bar_chart(stats.set_index("指标"))


def resolve_venv_python() -> Path:
  venv_python = ROOT / ".venv_tf310_dml" / "Scripts" / "python.exe"
  if not venv_python.is_file():
    raise FileNotFoundError(f"未找到虚拟环境 Python: {venv_python}")
  return venv_python


def init_file_browser_state() -> None:
  if "file_browser" not in st.session_state:
    st.session_state.file_browser = {
        "open": False,
        "target": None,
        "path": str(ROOT),
    }
  if "picked_path" not in st.session_state:
    st.session_state.picked_path = {}


def is_file_param(arg: dict[str, Any]) -> bool:
  key = str(arg["key"])
  label = str(arg.get("label", "")).lower()
  default = str(arg.get("default", ""))
  return any(token in key.lower() or token in label or token in default.lower() for token in ("csv", "pcap", "output", "metadata", "scaler", "npz", "dir"))


def open_file_browser(target_key: str, current_value: str = "") -> None:
  st.session_state.file_browser = {
      "open": True,
      "target": target_key,
      "path": current_value or str(ROOT),
  }


def close_file_browser() -> None:
  st.session_state.file_browser = {
      "open": False,
      "target": None,
      "path": st.session_state.file_browser.get("path", str(ROOT)),
  }


def list_directory(path: Path) -> tuple[list[Path], list[Path]]:
  dirs: list[Path] = []
  files: list[Path] = []
  try:
    for item in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
      if item.is_dir():
        dirs.append(item)
      else:
        files.append(item)
  except Exception:
    pass
  return dirs, files


st.set_page_config(page_title="Federated Flow Detection", layout="wide")
st.title("流量检测管理界面")
st.caption(" ")

try:
  VENV_PYTHON = resolve_venv_python()
  VENV_OK = True
except Exception as exc:
  VENV_PYTHON = Path(sys.executable)
  VENV_OK = False
  VENV_ERR = str(exc)

drain_logs()
state: TaskState = st.session_state.task
init_file_browser_state()

with st.sidebar:
  st.header("功能区")
  st.write("通过上方选项卡切换功能模块。")
  st.divider()
  st.write("运行说明")
  st.write("任务在后台执行，结果和日志会显示在主界面下方。")
  if VENV_OK:
    st.success(f"虚拟环境已就绪：{VENV_PYTHON}")
  else:
    st.error(f"虚拟环境不可用：{VENV_ERR}")

main_tabs = st.tabs([spec["label"] for spec in GUI_TASKS.values()])
selected_task_id = list(GUI_TASKS.keys())[0]
for idx, task_id in enumerate(GUI_TASKS.keys()):
  with main_tabs[idx]:
    selected_task_id = task_id
    spec = GUI_TASKS[task_id]
    params: dict[str, Any] = {}
    main_col, side_col = st.columns([1.2, 1])
    with main_col:
      sub_tabs = st.tabs(["参数", "帮助"])
      with sub_tabs[0]:
        st.write("参数配置")
        ignore_warnings = st.checkbox("忽略警告", value=False, key=f"{task_id}:ignore_warnings")
        for i, arg in enumerate(spec["args"]):
          key = str(arg["key"])
          label = str(arg["label"])
          typ = str(arg["type"])
          default = arg.get("default")
          widget_key = f"{task_id}:{key}:{i}"
          if typ == "checkbox":
            params[key] = st.checkbox(label, value=bool(default), key=widget_key)
          elif typ == "select":
            options = [str(x) for x in arg["options"]]
            params[key] = st.selectbox(label, options=options, index=options.index(str(default)), key=widget_key)
          elif typ == "number":
            step = arg.get("step", 1)
            if isinstance(default, int) and not isinstance(default, bool):
              params[key] = int(st.number_input(label, value=int(default), step=int(step), key=widget_key))
            else:
              params[key] = float(st.number_input(label, value=float(default), step=float(step), key=widget_key))
          else:
            is_path_like = is_file_param(arg)
            current_value = st.session_state.get("picked_path", {}).get(widget_key, str(default))
            if is_path_like:
              params[key] = st.text_input(label, value=str(current_value), key=widget_key)
              uploaded = st.file_uploader(
                  f"选择 {label}",
                  key=f"upload:{widget_key}",
                  label_visibility="collapsed",
                  accept_multiple_files=False,
              )
              if uploaded is not None:
                target_dir = Path(tempfile.gettempdir()) / "fl_streamlit_uploads"
                target_dir.mkdir(parents=True, exist_ok=True)
                safe_name = uploaded.name.replace("\\", "_").replace("/", "_")
                target_path = target_dir / safe_name
                with open(target_path, "wb") as f:
                  f.write(uploaded.getbuffer())
                params[key] = str(target_path)
                st.session_state.picked_path[widget_key] = str(target_path)
                st.success(f"文件已选择：{uploaded.name}")
            else:
              params[key] = st.text_input(label, value=str(default), key=widget_key)

        if selected_task_id == "train_cnn":
          st.info("如需使用 TensorFlow Federated，请先在系统环境中设置 `FL_USE_TFF=1` 再启动界面。")
        if selected_task_id == "infer_csv":
          st.warning("请使用与当前模型版本匹配的真实 CSV 数据，并确保 artifacts 与 processed 目录来自同一训练流程。")

      with sub_tabs[1]:
        st.write("功能说明")
        if task_id == "prepare":
          st.write("- 生成 `data/processed` 下的客户端 NPZ、metadata 和 scaler")
          st.write("- 支持留出测试集、BENIGN 下采样和 demo 数据")
          st.write("- 适合把 CICIDS2017 或相似 CSV 转成联邦训练数据")
        elif task_id == "train_cnn":
          st.write("- 进行 10 客户端联邦训练")
          st.write("- 输出 `global_model.keras`、`metadata.json`、`global_scaler.npz`")
        elif task_id == "eval_cnn":
          st.write("- 读取 NPZ 进行多分类与异常检测评估")
          st.write("- 可查看混淆矩阵和召回率")
        else:
          st.write("- 对单个或多个 CSV 逐行预测")
          st.write("- 支持 top-k 和概率显示")

    with side_col:
      pass


    run_disabled = state.running
    if st.button("开始执行", type="primary", disabled=run_disabled, key=f"run:{task_id}"):
      summary_path = Path(tempfile.gettempdir()) / f"fl_{task_id}_summary.json"
      if summary_path.exists():
        try:
          summary_path.unlink()
        except Exception:
          pass
      command = build_task_command(task_id, params, VENV_PYTHON, summary_path)
      launch_command(command, ROOT, ignore_warnings=ignore_warnings)
      st.session_state[f"summary_path:{task_id}"] = str(summary_path)

    status_box = st.container(border=True)
    log_box = st.container(border=True)
    log_placeholder = log_box.empty()
    status_placeholder = status_box.empty()

    if state.running:
      status_placeholder.success("任务正在运行，日志将自动刷新。")
      while state.running:
        drain_logs()
        log_text = "\n".join(state.log[-400:]) if state.log else ""
        if log_text:
          log_placeholder.code(log_text, language="text")
        else:
          log_placeholder.caption("暂无日志。")
        threading.Event().wait(0.6)
        if state.process is not None and state.process.poll() is not None:
          state.running = False
          state.returncode = state.process.returncode
          drain_logs()
          break
      drain_logs()
      status_placeholder.info(f"任务已结束，退出码：{state.returncode}")
    else:
      if state.returncode is not None:
        status_placeholder.info(f"上一次任务已结束，退出码：{state.returncode}")
      log_text = "\n".join(state.log[-400:]) if state.log else ""
      if log_text:
        log_placeholder.code(log_text, language="text")
      else:
        log_placeholder.caption("暂无日志。启动任务后，这里会显示脚本输出。")

    summary_path_value = st.session_state.get(f"summary_path:{task_id}")
    if summary_path_value:
      summary = read_json_if_exists(Path(summary_path_value))
      if summary:
        st.divider()
        st.subheader("结果图表")
        show_summary_charts(task_id, summary)
