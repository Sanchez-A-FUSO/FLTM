"""Lightweight PCAP -> CICIDS2017-style flow feature exporter.

This script is a dependency-light alternative to CICFlowMeter. It reads a PCAP
/ PCAPNG file, groups packets into bi-directional 5-tuple flows, and exports a
CSV whose columns follow the feature names listed in `1.csv`.

What is computed exactly:
- flow duration
- forward/backward packet counts and byte counts
- packet length statistics
- IAT statistics
- TCP flag counts
- basic header-length approximations
- a few simple rate / subflow metrics

What is approximated or left as zero:
- bulk-related metrics
- CWE / ECE counts in most captures
- exact active/idle segmentation
- some TCP-specific internal counters that require a full flow engine

Typical usage:

  python scripts/pcap_to_simple_flow_csv.py \
    --pcap capture.pcap \
    --output flows.csv \
    --label BENIGN

If you have a training metadata file, you can also compare the produced
columns against the expected feature names:

  python scripts/pcap_to_simple_flow_csv.py \
    --pcap capture.pcap \
    --output flows.csv \
    --metadata data/processed/metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Iterator, Optional

import numpy as np


CSV_COLUMNS = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]


@dataclass
class PacketRecord:
  ts: float
  src: str
  dst: str
  sport: int
  dport: int
  proto: str
  length: int
  payload_len: int = 0
  tcp_flags: str = ""
  ip_header_len: int = 0
  tcp_header_len: int = 0


@dataclass
class FlowStats:
  packets: list[PacketRecord] = field(default_factory=list)
  fwd_packets: list[PacketRecord] = field(default_factory=list)
  bwd_packets: list[PacketRecord] = field(default_factory=list)
  first_seen: float = math.inf
  last_seen: float = -math.inf

  def add(self, pkt: PacketRecord, forward: bool) -> None:
    self.packets.append(pkt)
    if forward:
      self.fwd_packets.append(pkt)
    else:
      self.bwd_packets.append(pkt)
    self.first_seen = min(self.first_seen, pkt.ts)
    self.last_seen = max(self.last_seen, pkt.ts)


def _safe_mean(values: list[float]) -> float:
  return float(mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
  return float(pstdev(values)) if len(values) >= 2 else 0.0


def _safe_min(values: list[float]) -> float:
  return float(min(values)) if values else 0.0


def _safe_max(values: list[float]) -> float:
  return float(max(values)) if values else 0.0


def _safe_sum(values: list[float]) -> float:
  return float(sum(values)) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
  if not values:
    return 0.0
  return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _count_flag(flag_texts: list[str], needle: str) -> int:
  needle = needle.upper()
  return int(sum(1 for f in flag_texts if needle in str(f).upper()))


def _count_cwe_ece(flag_texts: list[str], kind: str) -> int:
  """Best-effort count for CICIDS columns that are not directly exposed by raw PCAPs.

  In many captures these flags are not available as separate TCP bits. We
  therefore approximate them as zero unless a packet dissection layer exposes
  them explicitly.
  """
  kind = kind.upper()
  if kind not in {"CWE", "ECE"}:
    return 0
  return 0


def _get_packet_reader(pcap_path: Path) -> Iterator[PacketRecord]:
  """Yield PacketRecord objects using scapy, with pyshark as fallback."""
  try:
    from scapy.all import PcapReader  # type: ignore
    from scapy.layers.inet import IP, TCP, UDP  # type: ignore

    with PcapReader(str(pcap_path)) as reader:
      for pkt in reader:
        if IP not in pkt:
          continue
        ip = pkt[IP]
        proto = int(ip.proto)
        proto_name = {6: "TCP", 17: "UDP"}.get(proto, str(proto))

        sport = dport = 0
        flags = ""
        tcp_hlen = 0
        if TCP in pkt:
          sport = int(pkt[TCP].sport)
          dport = int(pkt[TCP].dport)
          flags = str(pkt[TCP].flags)
          try:
            tcp_hlen = int(pkt[TCP].dataofs) * 4 if pkt[TCP].dataofs is not None else 20
          except Exception:
            tcp_hlen = 20
        elif UDP in pkt:
          sport = int(pkt[UDP].sport)
          dport = int(pkt[UDP].dport)
        else:
          continue

        try:
          ip_hlen = int(ip.ihl) * 4 if ip.ihl is not None else 20
        except Exception:
          ip_hlen = 20

        length = int(len(pkt))
        payload_len = max(0, length - ip_hlen - tcp_hlen)
        yield PacketRecord(
            ts=float(pkt.time),
            src=str(ip.src),
            dst=str(ip.dst),
            sport=sport,
            dport=dport,
            proto=proto_name,
            length=length,
            payload_len=payload_len,
            tcp_flags=flags,
            ip_header_len=ip_hlen,
            tcp_header_len=tcp_hlen,
        )
    return
  except Exception:
    pass

  try:
    import pyshark  # type: ignore
  except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Neither scapy nor pyshark is available. Install one of them to parse PCAP files."
    ) from e

  cap = pyshark.FileCapture(str(pcap_path), keep_packets=False)
  try:
    for pkt in cap:
      if not hasattr(pkt, "ip"):
        continue
      ip = pkt.ip
      proto_name = getattr(pkt, "transport_layer", "") or ""
      if proto_name not in {"TCP", "UDP"}:
        continue
      l4 = getattr(pkt, proto_name.lower(), None)
      if l4 is None:
        continue
      sport = int(getattr(l4, "srcport", 0))
      dport = int(getattr(l4, "dstport", 0))
      flags = str(getattr(l4, "flags", "")) if proto_name == "TCP" else ""
      length = int(getattr(pkt, "length", 0) or 0)
      ts = float(pkt.sniff_timestamp)
      ip_hlen = 20
      tcp_hlen = 20 if proto_name == "TCP" else 8
      payload_len = max(0, length - ip_hlen - tcp_hlen)
      yield PacketRecord(ts, str(ip.src), str(ip.dst), sport, dport, proto_name, length, payload_len, flags, ip_hlen, tcp_hlen)
  finally:
    cap.close()


def _flow_key(pkt: PacketRecord) -> tuple[str, str, int, int, str]:
  a = (pkt.src, pkt.sport)
  b = (pkt.dst, pkt.dport)
  if a <= b:
    return pkt.src, pkt.dst, pkt.sport, pkt.dport, pkt.proto
  return pkt.dst, pkt.src, pkt.dport, pkt.sport, pkt.proto


def _build_flows(packets: Iterable[PacketRecord]) -> dict[tuple[str, str, int, int, str], FlowStats]:
  flows: dict[tuple[str, str, int, int, str], FlowStats] = {}
  for pkt in packets:
    key = _flow_key(pkt)
    stats = flows.setdefault(key, FlowStats())
    canonical_src, canonical_dst, canonical_sport, canonical_dport, _ = key
    forward = (pkt.src, pkt.sport) == (canonical_src, canonical_sport)
    stats.add(pkt, forward=forward)
  return flows


def _flow_active_idle(times: list[float]) -> dict[str, float]:
  """Approximate active/idle stats from inter-packet gaps.

  We define a new active segment when the gap between packets is <= 2 seconds.
  Larger gaps are treated as idle periods.
  """
  if len(times) < 2:
    return {
        "Active Mean": 0.0,
        "Active Std": 0.0,
        "Active Max": 0.0,
        "Active Min": 0.0,
        "Idle Mean": 0.0,
        "Idle Std": 0.0,
        "Idle Max": 0.0,
        "Idle Min": 0.0,
    }

  gaps = [t2 - t1 for t1, t2 in zip(times, times[1:])]
  active_segments: list[float] = []
  idle_segments: list[float] = []
  current_active = 0.0

  for gap in gaps:
    if gap <= 2.0:
      current_active += gap
    else:
      if current_active > 0:
        active_segments.append(current_active)
      idle_segments.append(gap)
      current_active = 0.0
  if current_active > 0:
    active_segments.append(current_active)

  return {
      "Active Mean": _safe_mean(active_segments),
      "Active Std": _safe_std(active_segments),
      "Active Max": _safe_max(active_segments),
      "Active Min": _safe_min(active_segments),
      "Idle Mean": _safe_mean(idle_segments),
      "Idle Std": _safe_std(idle_segments),
      "Idle Max": _safe_max(idle_segments),
      "Idle Min": _safe_min(idle_segments),
  }


def _flow_features(key: tuple[str, str, int, int, str], stats: FlowStats, label: str) -> dict[str, object]:
  src, dst, sport, dport, proto = key
  packets = stats.packets
  fwd = stats.fwd_packets
  bwd = stats.bwd_packets

  times = [p.ts for p in packets]
  lengths = [float(p.length) for p in packets]
  fwd_lengths = [float(p.length) for p in fwd]
  bwd_lengths = [float(p.length) for p in bwd]
  fwd_payloads = [float(p.payload_len) for p in fwd]
  bwd_payloads = [float(p.payload_len) for p in bwd]
  fwd_hdrs = [float(p.ip_header_len + p.tcp_header_len) for p in fwd]
  bwd_hdrs = [float(p.ip_header_len + p.tcp_header_len) for p in bwd]

  iats = [t2 - t1 for t1, t2 in zip(times, times[1:])]
  fwd_times = [p.ts for p in fwd]
  bwd_times = [p.ts for p in bwd]
  fwd_iats = [t2 - t1 for t1, t2 in zip(fwd_times, fwd_times[1:])]
  bwd_iats = [t2 - t1 for t1, t2 in zip(bwd_times, bwd_times[1:])]

  duration = max(0.0, stats.last_seen - stats.first_seen)
  total_fwd = len(fwd)
  total_bwd = len(bwd)
  total_packets = len(packets)
  total_fwd_bytes = int(sum(fwd_lengths))
  total_bwd_bytes = int(sum(bwd_lengths))
  total_bytes = total_fwd_bytes + total_bwd_bytes

  flag_texts = [p.tcp_flags for p in packets if p.proto == "TCP"]
  fwd_flag_texts = [p.tcp_flags for p in fwd if p.proto == "TCP"]
  bwd_flag_texts = [p.tcp_flags for p in bwd if p.proto == "TCP"]

  flow_bytes_s = (total_bytes / duration) if duration > 0 else float(total_bytes)
  flow_pkts_s = (total_packets / duration) if duration > 0 else float(total_packets)
  fwd_pkts_s = (total_fwd / duration) if duration > 0 else float(total_fwd)
  bwd_pkts_s = (total_bwd / duration) if duration > 0 else float(total_bwd)

  min_pkt_len = _safe_min(lengths)
  max_pkt_len = _safe_max(lengths)
  avg_pkt_size = _safe_mean(lengths)

  def _ratio(a: float, b: float) -> float:
    return float(a / b) if b else 0.0

  active_idle = _flow_active_idle(times)

  row: dict[str, object] = {
      "Destination Port": dport,
      "Flow Duration": duration,
      "Total Fwd Packets": total_fwd,
      "Total Backward Packets": total_bwd,
      "Total Length of Fwd Packets": total_fwd_bytes,
      "Total Length of Bwd Packets": total_bwd_bytes,
      "Fwd Packet Length Max": _safe_max(fwd_lengths),
      "Fwd Packet Length Min": _safe_min(fwd_lengths),
      "Fwd Packet Length Mean": _safe_mean(fwd_lengths),
      "Fwd Packet Length Std": _safe_std(fwd_lengths),
      "Bwd Packet Length Max": _safe_max(bwd_lengths),
      "Bwd Packet Length Min": _safe_min(bwd_lengths),
      "Bwd Packet Length Mean": _safe_mean(bwd_lengths),
      "Bwd Packet Length Std": _safe_std(bwd_lengths),
      "Flow Bytes/s": flow_bytes_s,
      "Flow Packets/s": flow_pkts_s,
      "Flow IAT Mean": _safe_mean(iats),
      "Flow IAT Std": _safe_std(iats),
      "Flow IAT Max": _safe_max(iats),
      "Flow IAT Min": _safe_min(iats),
      "Fwd IAT Total": _safe_sum(fwd_iats),
      "Fwd IAT Mean": _safe_mean(fwd_iats),
      "Fwd IAT Std": _safe_std(fwd_iats),
      "Fwd IAT Max": _safe_max(fwd_iats),
      "Fwd IAT Min": _safe_min(fwd_iats),
      "Bwd IAT Total": _safe_sum(bwd_iats),
      "Bwd IAT Mean": _safe_mean(bwd_iats),
      "Bwd IAT Std": _safe_std(bwd_iats),
      "Bwd IAT Max": _safe_max(bwd_iats),
      "Bwd IAT Min": _safe_min(bwd_iats),
      "Fwd PSH Flags": _count_flag(fwd_flag_texts, "P"),
      "Bwd PSH Flags": _count_flag(bwd_flag_texts, "P"),
      "Fwd URG Flags": _count_flag(fwd_flag_texts, "U"),
      "Bwd URG Flags": _count_flag(bwd_flag_texts, "U"),
      "Fwd Header Length": _safe_sum(fwd_hdrs),
      "Bwd Header Length": _safe_sum(bwd_hdrs),
      "Fwd Packets/s": fwd_pkts_s,
      "Bwd Packets/s": bwd_pkts_s,
      "Min Packet Length": min_pkt_len,
      "Max Packet Length": max_pkt_len,
      "Packet Length Mean": _safe_mean(lengths),
      "Packet Length Std": _safe_std(lengths),
      "Packet Length Variance": float(np.var(lengths, dtype=np.float64)) if lengths else 0.0,
      "FIN Flag Count": _count_flag(flag_texts, "F"),
      "SYN Flag Count": _count_flag(flag_texts, "S"),
      "RST Flag Count": _count_flag(flag_texts, "R"),
      "PSH Flag Count": _count_flag(flag_texts, "P"),
      "ACK Flag Count": _count_flag(flag_texts, "A"),
      "URG Flag Count": _count_flag(flag_texts, "U"),
      "CWE Flag Count": _count_cwe_ece(flag_texts, "CWE"),
      "ECE Flag Count": _count_cwe_ece(flag_texts, "ECE"),
      "Down/Up Ratio": _ratio(total_bwd, total_fwd),
      "Average Packet Size": avg_pkt_size,
      "Avg Fwd Segment Size": _safe_mean(fwd_payloads),
      "Avg Bwd Segment Size": _safe_mean(bwd_payloads),
      "Fwd Avg Bytes/Bulk": 0.0,
      "Fwd Avg Packets/Bulk": 0.0,
      "Fwd Avg Bulk Rate": 0.0,
      "Bwd Avg Bytes/Bulk": 0.0,
      "Bwd Avg Packets/Bulk": 0.0,
      "Bwd Avg Bulk Rate": 0.0,
      "Subflow Fwd Packets": total_fwd,
      "Subflow Fwd Bytes": total_fwd_bytes,
      "Subflow Bwd Packets": total_bwd,
      "Subflow Bwd Bytes": total_bwd_bytes,
      "Init_Win_bytes_forward": 0,
      "Init_Win_bytes_backward": 0,
      "act_data_pkt_fwd": max(0, total_fwd - 1),
      "min_seg_size_forward": int(min(fwd_payloads) if fwd_payloads else 0),
  }
  row.update(active_idle)
  row["Label"] = label
  return row


def _write_csv(rows: list[dict[str, object]], output: Path) -> None:
  if not rows:
    raise SystemExit("No flows were extracted from the PCAP.")
  output.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = CSV_COLUMNS + ["Label"]
  with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({name: row.get(name, 0) for name in fieldnames})


def _load_metadata(meta_path: Path) -> dict[str, object]:
  with open(meta_path, "r", encoding="utf-8") as f:
    return json.load(f)


def main() -> None:
  parser = argparse.ArgumentParser(description="Convert PCAP to CICIDS2017-style CSV features")
  parser.add_argument("--pcap", required=True, help="Input .pcap/.pcapng file")
  parser.add_argument("--output", required=True, help="Output CSV path")
  parser.add_argument("--label", default="BENIGN", help="Constant label written to each flow")
  parser.add_argument(
      "--metadata",
      default="",
      help="Optional metadata.json from training; used only to report overlap with known features",
  )
  parser.add_argument(
      "--limit",
      type=int,
      default=0,
      help="Optional maximum number of packets to read (0 = all)",
  )
  args = parser.parse_args()

  pcap_path = Path(args.pcap)
  if not pcap_path.is_file():
    raise SystemExit(f"PCAP not found: {pcap_path}")

  packets: list[PacketRecord] = []
  for i, pkt in enumerate(_get_packet_reader(pcap_path), start=1):
    packets.append(pkt)
    if args.limit > 0 and i >= args.limit:
      break

  flows = _build_flows(packets)
  rows = [_flow_features(key, stats, args.label) for key, stats in flows.items()]
  rows.sort(key=lambda r: (str(r["Destination Port"]), str(r["Flow Duration"]), str(r["Label"])))
  _write_csv(rows, Path(args.output))

  print(f"Extracted {len(packets)} packets -> {len(rows)} flows")
  print(f"Wrote {args.output}")

  if args.metadata:
    meta_path = Path(args.metadata)
    if meta_path.is_file():
      meta = _load_metadata(meta_path)
      training_features = list(meta.get("feature_names", []))
      produced = set(CSV_COLUMNS)
      overlap = [c for c in training_features if c in produced]
      missing = [c for c in training_features if c not in produced]
      print(f"Metadata feature overlap: {len(overlap)}/{len(training_features)}")
      if missing:
        print("Missing training features (first 20):")
        print(", ".join(missing[:20]))
    else:
      print(f"Metadata file not found: {meta_path}")


if __name__ == "__main__":
  main()
