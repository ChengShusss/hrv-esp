"""Raw/filtered ECG viewer with provisional R peaks and RR intervals."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ecg_protocol import Frame, LineDecoder, Waveform, encode_frame
from ecg_processing import EcgProcessor
from ecg_recording import RecordingSession, analyze_recording
from hrv_view import HrvResults

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


class EcgApp:
    def __init__(self, root: tk.Tk, port: str | None = None, demo: bool = False):
        self.root = root
        self.connection = None
        self.demo = demo
        self.demo_start = time.monotonic()
        self.demo_index = 0
        self.decoder = LineDecoder()
        self.waveform = Waveform()
        self.processor = EcgProcessor()
        self.last_received = 0.0
        self.sync_until = None
        self.synchronized = True
        self.recording = None
        self.analysis_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hrv-analysis")
        self.analysis_future = None
        self.pending_show_result = True
        self.latest_result = None
        self.result_window = None
        self.data_paused = False
        self.record_error = ""
        self.closed = False
        self.tick_id = None

        root.title("ESP32-C3 · AD8232 心电波形")
        root.geometry("1180x820")
        root.minsize(1000, 720)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.port_var = tk.StringVar(value=port or "")
        self.status_var = tk.StringVar(value="未连接 — 选择开发板的 USB 串口")
        self.metrics_var = tk.StringVar(value="原始 ADC 数据 · 等待采样")
        self.window_var = tk.StringVar(value="5")
        self.autoscale_var = tk.BooleanVar(value=True)
        self.polarity_var = tk.StringVar(value="向上")
        self.analysis_var = tk.StringVar(value="等待采样 · RR --")
        self.rr_var = tk.StringVar(value="最近 RR（ms）：--")
        self.recording_var = tk.StringVar(value="开始记录后保留原始数据和 RR；连续有效 RR 达 5 分钟可计算 LF/HF。")

        controls = ttk.Frame(root, padding=(16, 12))
        controls.pack(fill="x")
        ttk.Label(controls, text="串口").pack(side="left", padx=(0, 8))
        self.port_box = ttk.Combobox(controls, textvariable=self.port_var, width=13,
                                     state="readonly")
        self.port_box.pack(side="left")
        self.refresh_button = ttk.Button(controls, text="刷新", command=self.refresh_ports)
        self.refresh_button.pack(side="left", padx=6)
        self.connect_button = ttk.Button(controls, text="连接", command=self.toggle_connection)
        self.connect_button.pack(side="left")
        ttk.Label(controls, text="显示最近").pack(side="left", padx=(16, 6))
        ttk.Combobox(controls, textvariable=self.window_var, values=("2", "5", "10"),
                     width=3, state="readonly").pack(side="left")
        ttk.Label(controls, text="秒").pack(side="left", padx=(4, 12))
        ttk.Checkbutton(controls, text="纵轴自动缩放", variable=self.autoscale_var).pack(side="left")

        recording_controls = ttk.Frame(root, padding=(16, 0))
        recording_controls.pack(fill="x")
        self.record_button = ttk.Button(recording_controls, text="开始记录…", command=self.choose_recording,
                                        state="disabled")
        self.record_button.pack(side="left")
        self.stop_record_button = ttk.Button(recording_controls, text="结束记录并计算 HRV",
                                             command=self.stop_recording, state="disabled")
        self.stop_record_button.pack(side="left", padx=8)
        self.result_button = ttk.Button(recording_controls, text="查看上次结果", command=self.show_result, state="disabled")
        self.result_button.pack(side="left")
        ttk.Label(root, textvariable=self.recording_var, padding=(18, 6), wraplength=940).pack(anchor="w")

        ttk.Label(root, textvariable=self.status_var, padding=(18, 4)).pack(anchor="w")
        ttk.Label(root, textvariable=self.metrics_var, padding=(18, 4)).pack(anchor="w")
        analysis_controls = ttk.Frame(root, padding=(18, 4))
        analysis_controls.pack(fill="x")
        ttk.Label(analysis_controls, text="R 峰方向").pack(side="left", padx=(0, 6))
        polarity_box = ttk.Combobox(analysis_controls, textvariable=self.polarity_var,
                                    values=("向上", "向下"), width=5, state="readonly")
        polarity_box.pack(side="left")
        polarity_box.bind("<<ComboboxSelected>>", self.change_polarity)
        ttk.Label(analysis_controls, textvariable=self.analysis_var).pack(side="left", padx=16)
        ttk.Label(root, textvariable=self.rr_var, padding=(18, 4)).pack(anchor="w")
        self.canvas = tk.Canvas(root, bg="#101827", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=16, pady=(8, 10))
        ttk.Label(root, text="R 峰回定位已抑制 50 Hz，红点对应原始采样时刻；RR 按采样序号计算。"
                  "下图为因果滤波输出，有相位延迟。检测结果仍需核验。",
                  padding=(18, 2)).pack(anchor="w")
        ttk.Label(root, text="人体电极连接期间，有线调试需适合人体连接的电气隔离；"
                  "普通 USB 联调请先取下电极。", padding=(18, 2)).pack(anchor="w", pady=(0, 10))
        self.refresh_ports()
        if demo:
            root.title("ESP32-C3 · AD8232 心电波形 [模拟演示]")
            self.status_var.set("模拟演示 — 合成波形，不是实际采集数据")
            self.connect_button.configure(state="disabled")
            self.port_box.configure(state="disabled")
            self.refresh_button.configure(state="disabled")
        elif port:
            self.port_var.set(port)
            self.connect()
        self.refresh_recording_controls()
        self.tick_id = root.after(30, self.tick)

    def refresh_ports(self):
        ports = list(list_ports.comports()) if list_ports is not None else []
        self.port_box["values"] = [item.device for item in ports]
        if not self.port_var.get() or self.port_var.get() not in self.port_box["values"]:
            preferred = next((item for item in ports if item.vid == 0x303A), None)
            self.port_var.set(preferred.device if preferred else "")

    def change_polarity(self, event=None):
        self.processor.set_polarity(1 if self.polarity_var.get() == "向上" else -1)

    def connect(self):
        if serial is None:
            messagebox.showerror("缺少串口依赖", "请运行：python -m pip install -r host/requirements.txt")
            return
        port = self.port_var.get()
        if not port:
            self.status_var.set("未找到串口，请用数据线连接 SuperMini 后点击刷新")
            return
        connection = serial.Serial(port=None, baudrate=115200, timeout=0)
        # Do not intentionally reset the ESP32 or put it into download mode.
        connection.dtr = False
        connection.rts = False
        connection.port = port
        try:
            connection.open()
            if hasattr(connection, "set_buffer_size"):
                try:
                    connection.set_buffer_size(rx_size=65536, tx_size=4096)
                except (serial.SerialException, OSError):
                    pass  # Not all USB drivers implement buffer resizing.
            connection.reset_input_buffer()
        except (serial.SerialException, OSError) as error:
            connection.close()
            self.status_var.set(f"无法打开 {port}：{error}")
            return
        self.connection = connection
        self.root.title(f"ESP32-C3 · AD8232 心电波形 [{port}]")
        self.decoder = LineDecoder()
        self.waveform = Waveform()
        self.processor = EcgProcessor(1 if self.polarity_var.get() == "向上" else -1)
        self.last_received = 0.0
        self.data_paused = False
        self.sync_until = None
        self.synchronized = False
        self.connect_button.configure(text="断开")
        self.port_box.configure(state="disabled")
        self.refresh_button.configure(state="disabled")
        self.status_var.set(f"{port} 已打开，等待 ECG1 数据…")
        self.refresh_recording_controls()

    def disconnect(self, reason="已断开"):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.connect_button.configure(text="连接")
        self.root.title("ESP32-C3 · AD8232 心电波形 [未连接]")
        self.port_box.configure(state="readonly")
        self.refresh_button.configure(state="normal")
        self.stop_recording(reason="disconnected")
        self.processor.pause("已断开")
        self.status_var.set(reason)
        self.refresh_recording_controls()

    def toggle_connection(self):
        if self.connection is not None:
            self.disconnect()
        else:
            self.connect()

    def refresh_recording_controls(self):
        available = (self.connection is not None or self.demo) and not self.closed
        self.record_button.configure(state="normal" if available and self.recording is None and self.analysis_future is None else "disabled")
        self.stop_record_button.configure(state="normal" if self.recording is not None else "disabled")
        self.result_button.configure(state="normal" if self.latest_result is not None else "disabled")

    def choose_recording(self):
        directory = Path(__file__).resolve().parents[1] / "captures"
        directory.mkdir(exist_ok=True)
        filename = filedialog.asksaveasfilename(
            title="开始记录：选择新文件名，结束后自动生成 RR 和 HRV 报告", defaultextension=".csv",
            initialdir=directory, initialfile=time.strftime("ecg_%Y%m%d_%H%M%S.csv"),
            filetypes=[("CSV", "*.csv")])
        if not filename:
            return
        try:
            self.start_recording(Path(filename))
        except (OSError, RuntimeError) as error:
            messagebox.showerror("无法开始记录", str(error))

    def start_recording(self, path: Path):
        if self.recording is not None or self.analysis_future is not None:
            raise RuntimeError("请先结束当前记录并等待计算完成")
        if self.connection is None and not self.demo:
            raise RuntimeError("请先连接开发板")
        self.recording = RecordingSession(path, "demo" if self.demo else "serial")
        self.record_error = ""
        self.recording_var.set(f"正在记录：{self.recording.raw_path.name}")
        self.refresh_recording_controls()

    def stop_recording(self, reason="manual", show_result=True):
        if self.recording is None:
            return
        recording, self.recording = self.recording, None
        metadata = recording.close(reason)
        self.pending_show_result = show_result
        self.analysis_future = self.analysis_executor.submit(analyze_recording, metadata)
        self.recording_var.set("记录已结束，正在计算 HRV 并保存报告…")
        self.refresh_recording_controls()

    def poll_analysis(self):
        if self.analysis_future is None or not self.analysis_future.done():
            return
        future, self.analysis_future = self.analysis_future, None
        try:
            self.latest_result = future.result()
            if self.latest_result["save_error"]:
                self.recording_var.set("HRV 报告未能保存；请在结果窗口中另存。" + self.latest_result["save_error"])
            elif self.latest_result["report"].get("analysis_error"):
                self.recording_var.set("记录已结束，HRV 未计算：" + self.latest_result["report"]["analysis_error"])
            else:
                self.recording_var.set("HRV 报告已保存：" + str(self.latest_result["saved_path"]))
            if self.pending_show_result:
                self.show_result()
        except Exception as error:
            self.record_error = f"HRV 计算失败：{error}"
            self.recording_var.set(self.record_error + "；原始记录保留，可用命令行重新分析。")
        self.refresh_recording_controls()

    def show_result(self):
        if self.latest_result is None or self.closed:
            return
        if self.result_window is not None and self.result_window.winfo_exists():
            self.result_window.destroy()
        self.result_window = HrvResults(self.root, **self.latest_result)

    def ingest(self, data: bytes):
        invalid_before = self.decoder.invalid_packets
        frames = self.decoder.feed(data)
        now = time.monotonic()
        if not self.synchronized:
            # reset_input_buffer() clears the PC buffer, but the device can still
            # have old packets in its TX FIFO/queue. Drain for 0.5 s after the first
            # bytes arrive; losses before this viewing session are not its gaps.
            if self.sync_until is None:
                self.sync_until = now + 0.5
            if now < self.sync_until:
                self.status_var.set("串口已打开，正在排空旧缓存…")
                return
            self.waveform = Waveform()
            self.decoder.invalid_packets = 0
            self.synchronized = True
        if self.recording is not None:
            self.recording.bad_packets += max(0, self.decoder.invalid_packets - invalid_before)
        for frame in frames:
            if not self.waveform.accept(frame):
                continue
            peaks = self.processor.process_frame(frame)
            self.last_received = time.monotonic()
            self.data_paused = False
            if self.recording is not None:
                try:
                    self.recording.add_frame(frame, peaks, self.processor.polarity)
                except (OSError, ValueError) as error:
                    self.record_error = f"记录失败：{error}"
                    self.stop_recording(reason="write_error")
                    self.status_var.set(self.record_error)

    def generate_demo(self, now: float):
        target = int((now - self.demo_start) * 1000)
        # Limit catch-up after a suspended UI; the resulting sample gap is visible.
        if target - self.demo_index > 1000:
            self.demo_index = (target // 32 - 1) * 32
        while self.demo_index + 32 <= target:
            values = []
            for index in range(self.demo_index, self.demo_index + 32):
                seconds = index / 1000
                phase = seconds % 0.85
                value = (2050 + 45 * math.sin(2 * math.pi * 0.25 * seconds)
                         + 90 * math.exp(-((phase - 0.15) / 0.04) ** 2)
                         - 170 * math.exp(-((phase - 0.27) / 0.012) ** 2)
                         + 1050 * math.exp(-((phase - 0.30) / 0.010) ** 2)
                         - 250 * math.exp(-((phase - 0.33) / 0.015) ** 2)
                         + 230 * math.exp(-((phase - 0.52) / 0.07) ** 2)
                         + 5 * math.sin(2 * math.pi * 43 * seconds))
                values.append(round(value))
            self.ingest(encode_frame(Frame(0xDE001234, 0, self.demo_index, 1000, 0, tuple(values))))
            self.demo_index += 32

    def tick(self):
        if self.closed:
            return
        now = time.monotonic()
        if self.demo:
            self.generate_demo(now)
        elif self.connection is not None:
            try:
                waiting = self.connection.in_waiting
                if waiting:
                    self.ingest(self.connection.read(min(waiting, 65536)))
            except (serial.SerialException, OSError) as error:
                self.disconnect(f"串口连接中断：{error}")

        frame = self.waveform.last_frame
        if frame is not None:
            lead_status = ("未检测到电极脱落" if not frame.leads_off else
                           f"电极脱落：LO+={frame.leads_off & 1}，LO−={(frame.leads_off >> 1) & 1}")
            if self.demo:
                self.status_var.set("模拟演示 — 合成波形，不是实际采集数据")
            elif self.connection is not None:
                if now - self.last_received > 2:
                    self.status_var.set("数据已停止超过 2 秒，请检查供电、连接和固件")
                    self.processor.pause()
                    if not self.data_paused and self.recording is not None:
                        self.recording.pauses += 1
                    self.data_paused = True
                else:
                    self.status_var.set(f"{self.connection.port} 正在采集 · {lead_status}")
            recent = list(self.waveform.points)[-1000:]
            rail_count = sum(value is not None and (value <= 8 or value >= 4087)
                             for _, value in recent)
            rail_text = f" · 近量程边缘 {rail_count} 点，检查削顶" if rail_count else ""
            self.metrics_var.set(f"{frame.sample_rate} Hz · 已收 {self.waveform.received_samples:,} 点"
                f" · 缺失 {self.waveform.missing_samples:,} 点 · 重新分段 {self.waveform.restarts} 次"
                f" · 损坏包 {self.decoder.invalid_packets}{rail_text}"
                + (f" · {self.record_error}" if self.record_error else ""))
        peak = self.processor.last_peak
        latest_rr = f"{peak.rr_ms:.1f} ms" if peak is not None and peak.rr_ms is not None else "--"
        self.analysis_var.set(f"{self.processor.status} · R 峰 {self.processor.detected_count} 个"
                              f" · 最新 RR {latest_rr}")
        intervals = [f"{item.rr_ms:.0f}" for item in self.processor.peaks if item.rr_ms is not None]
        self.rr_var.set("最近 RR（历史 / ms）：" + (" · ".join(intervals[-10:]) or "--"))
        if self.recording is not None:
            recording = self.recording
            elapsed = int(now - recording.started_clock)
            support = recording.frequency_support_s if peak is not None and recording.last_peak == peak else 0.0
            self.recording_var.set(f"记录中 {elapsed//60:02d}:{elapsed%60:02d} · {recording.samples:,} 点 · "
                                   f"连续有效 RR {support:.1f} / 300 秒 · {recording.rr_count} 个 RR · "
                                   f"脱落/削顶样本 {recording.invalid_samples}")
        self.poll_analysis()
        self.draw()
        self.tick_id = self.root.after(30, self.tick)

    def draw(self):
        canvas = self.canvas
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 100 or height < 100:
            return
        duration = float(self.window_var.get())
        points = list(self.waveform.points)
        end = points[-1][0] if points else 0.0
        start = end - duration
        points = [(stamp, value) for stamp, value in points if stamp >= start]
        filtered = [(stamp, value) for stamp, value in self.processor.filtered if start <= stamp <= end]
        middle = height // 2
        panels = [(points, 24, middle - 30, "原始 ADC · 红点为 R 峰", False),
                  (filtered, middle + 26, height - 34, "滤波：0.5–40 Hz，50 Hz 陷波", True)]
        left, right = 66, width - 24
        for trace, top, bottom, label, signed in panels:
            values = [value for _, value in trace if value is not None]
            low, high = (-4095.0, 4095.0) if signed else (0.0, 4095.0)
            if values and self.autoscale_var.get():
                lo, hi = min(values), max(values)
                margin = max(30, (hi - lo) * 0.12)
                low, high = lo - margin, hi + margin
                if not signed:
                    low, high = max(0, low), min(4095, high)
            for index in range(5):
                y = top + index / 4 * (bottom - top)
                canvas.create_line(left, y, right, y, fill="#273449")
                canvas.create_text(left - 10, y, text=f"{high - index / 4 * (high - low):.0f}",
                                   fill="#9cacbf", anchor="e")
            for index in range(6):
                x = left + index / 5 * (right - left)
                canvas.create_line(x, top, x, bottom, fill="#273449")
                canvas.create_text(x, bottom + 15, text=f"{-duration + index / 5 * duration:g}s", fill="#9cacbf")
            canvas.create_text(left, top - 13, text=label, anchor="w", fill="#c2cedc")
            if not values:
                canvas.create_text((left + right) / 2, (top + bottom) / 2,
                                   text="等待有效数据…", fill="#9cacbf")
                continue
            color = "#56dfbb" if signed else "#9aaebf"
            if not signed and self.waveform.last_frame.leads_off:
                color = "#f3b85b"
            coordinates = []
            for stamp, value in trace:
                if value is None:
                    if len(coordinates) >= 4:
                        canvas.create_line(*coordinates, fill=color, width=1.4)
                    coordinates = []
                    continue
                coordinates.extend((left + (stamp - start) / duration * (right - left),
                                    bottom - (value - low) / (high - low) * (bottom - top)))
            if len(coordinates) >= 4:
                canvas.create_line(*coordinates, fill=color, width=1.4)
            if not signed:
                for peak in self.processor.peaks:
                    if start <= peak.time_s <= end:
                        x = left + (peak.time_s - start) / duration * (right - left)
                        y = bottom - (peak.raw_adc - low) / (high - low) * (bottom - top)
                        canvas.create_oval(x - 4, y - 4, x + 4, y + 4,
                                           fill="#ff6b78", outline="", tags=("r_peak",))
                        canvas.create_text(x, y - 12, text="R", fill="#ff6b78")

    def close(self):
        self.closed = True
        if self.tick_id is not None:
            self.root.after_cancel(self.tick_id)
        self.stop_recording(reason="window_closed", show_result=False)
        self.disconnect()
        self.analysis_executor.shutdown(wait=False)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="ESP32-C3 / AD8232 USB ECG waveform viewer")
    parser.add_argument("--port", help="Open a specific port, for example COM5")
    parser.add_argument("--demo", action="store_true", help="Show synthetic ECG without a device")
    parser.add_argument("--list-ports", action="store_true", help="List serial ports and exit")
    args = parser.parse_args()
    if args.demo and args.port:
        parser.error("--demo and --port cannot be used together")
    if args.list_ports:
        if list_ports is None:
            parser.error("Install pyserial: python -m pip install -r host/requirements.txt")
        for port in list_ports.comports():
            print(f"{port.device}: {port.description} [{port.hwid}]")
        return 0
    root = tk.Tk()
    EcgApp(root, port=args.port, demo=args.demo)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
