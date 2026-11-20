"""Read-only HRV result window; report calculations belong to hrv.py."""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ecg_recording import save_report_new


def number(value, places=2):
    return "—" if value is None else f"{value:.{places}f}"


class HrvResults(tk.Toplevel):
    def __init__(self, parent, report, saved_path=None, save_error=None):
        super().__init__(parent)
        self.report = report
        self.segments = report.get("segments", [])
        self.current = None
        self.title("HRV 记录结果")
        self.geometry("1120x820")
        self.minsize(940, 720)
        self.summary_var = tk.StringVar()
        self.notice_var = tk.StringVar()
        self.path_var = tk.StringVar(value=f"报告：{saved_path}" if saved_path else f"报告未保存：{save_error or '可另存'}")
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill="both", expand=True)
        record = report.get("recording", {})
        source = record.get("source", self.segments[0].get("source", "") if self.segments else "")
        source_text = "模拟数据（不是真实心电）" if source == "demo" else "实测 RR 估计"
        label = source_text + f" · 共 {len(self.segments)} 个独立分析段"
        if record:
            label += (f" · 记录 {record.get('wall_duration_s', 0):.1f} 秒 · {record.get('samples', 0):,} 点"
                      f" · 丢样 {record.get('missing_samples', 0)} · 损坏包 {record.get('bad_packets', 0)}"
                      f" · 脱落/削顶样本 {record.get('invalid_samples', 0)}")
            reason = {"manual": "手动结束", "disconnected": "断连结束", "window_closed": "关闭窗口结束",
                      "write_error": "写入失败"}.get(record.get("stop_reason"), record.get("stop_reason", ""))
            label += f" · {reason}"
        ttk.Label(frame, text=label, wraplength=880).pack(anchor="w", pady=(0, 8))
        selection = ttk.Frame(frame)
        selection.pack(fill="x")
        ttk.Label(selection, text="分析片段").pack(side="left", padx=(0, 8))
        self.segment_box = ttk.Combobox(selection, state="readonly", width=65)
        self.segment_box["values"] = [f"第 {i+1} 段 · {s['metrics']['rr_duration_s']:.1f} 秒 · "
                                      f"{s['metrics']['rr_count']} 个 RR · "
                                      + ("频域可计算" if s['frequency']['status'] == 'available_rr_estimate' else "频域待满足条件")
                                      for i, s in enumerate(self.segments)]
        self.segment_box.pack(side="left", fill="x", expand=True)
        self.segment_box.bind("<<ComboboxSelected>>", self.select_segment)
        ttk.Button(selection, text="另存报告…", command=self.save_as).pack(side="right", padx=(12, 0))
        ttk.Label(frame, textvariable=self.summary_var, wraplength=880).pack(anchor="w", pady=(8, 4))
        ttk.Label(frame, textvariable=self.notice_var, wraplength=880, foreground="#9a5800").pack(anchor="w", pady=(0, 8))
        metrics = ttk.Frame(frame)
        metrics.pack(fill="x")
        self.time_values, self.frequency_values = {}, {}
        time_box = ttk.LabelFrame(metrics, text="时域（基于 RR）", padding=10)
        freq_box = ttk.LabelFrame(metrics, text="频域（基于 RR）", padding=10)
        time_box.pack(side="left", fill="both", expand=True, padx=(0, 8))
        freq_box.pack(side="left", fill="both", expand=True)
        time_fields = [("mean_hr_bpm", "平均心率", "次/分"), ("mean_rr_ms", "平均 RR", "ms"),
                       ("rmssd_ms", "RMSSD", "ms"), ("sdrr_ms", "SDRR（RR 标准差）", "ms"),
                       ("ln_rmssd", "ln(RMSSD)", ""), ("pnn50_percent", "pNN50", "%")]
        frequency_fields = [("lf_ms2", "LF · 0.04–0.15 Hz", "ms²"), ("hf_ms2", "HF · 0.15–0.40 Hz", "ms²"),
                            ("lf_hf_ratio", "LF/HF", ""), ("lf_nu_percent", "归一化 LF", "%"),
                            ("hf_nu_percent", "归一化 HF", "%"), ("lf_hf_sum_ms2", "LF + HF 功率", "ms²")]
        for box, definitions, variables in ((time_box, time_fields, self.time_values),
                                             (freq_box, frequency_fields, self.frequency_values)):
            box.columnconfigure(1, weight=1)
            for row, (key, name, units) in enumerate(definitions):
                ttk.Label(box, text=name).grid(row=row, column=0, sticky="w", padx=(0, 24), pady=3)
                variable = tk.StringVar(value="—")
                variables[key] = variable
                ttk.Label(box, textvariable=variable).grid(row=row, column=1, sticky="e", padx=(0, 8))
                ttk.Label(box, text=units, width=6).grid(row=row, column=2, sticky="w")
        self.tabs = ttk.Notebook(frame)
        self.tabs.pack(fill="both", expand=True, pady=(12, 8))
        self.rr_canvas = tk.Canvas(self.tabs, bg="#101827", highlightthickness=0)
        self.psd_canvas = tk.Canvas(self.tabs, bg="#101827", highlightthickness=0)
        self.tabs.add(self.rr_canvas, text="RR 间隔")
        self.tabs.add(self.psd_canvas, text="LF / HF 功率谱")
        self.rr_canvas.bind("<Configure>", lambda event: self.draw())
        self.psd_canvas.bind("<Configure>", lambda event: self.draw())
        self.tabs.bind("<<NotebookTabChanged>>", lambda event: self.draw())
        ttk.Label(frame, text="各段独立计算，未进行正常/异位搏动分类。LF/HF 不作为压力评分或交感/副交感平衡判断。",
                  wraplength=880).pack(anchor="w", pady=(0, 4))
        ttk.Label(frame, textvariable=self.path_var, wraplength=880).pack(anchor="w")
        if self.segments:
            best = max(range(len(self.segments)), key=lambda i: (
                self.segments[i]["frequency"]["status"] == "available_rr_estimate",
                self.segments[i]["metrics"]["rr_duration_s"]))
            self.segment_box.current(best)
            self.select_segment()
        else:
            self.summary_var.set("本次记录没有可计算的 RR 数据")
            self.notice_var.set(report.get("analysis_error", "需要连续采集到至少两个 R 峰；LF/HF 需要更长记录。"))

    def select_segment(self, event=None):
        selection = self.segment_box.current()
        if selection < 0:
            return
        self.current = self.segments[selection]
        metrics, frequency = self.current["metrics"], self.current["frequency"]
        for key, variable in self.time_values.items():
            variable.set(number(metrics[key], 3 if key == "ln_rmssd" else 2))
        for key, variable in self.frequency_values.items():
            variable.set(number(frequency[key]))
        self.summary_var.set(f"本段 {self.current['peak_count']} 个 R 峰、{metrics['rr_count']} 个 RR，"
                             f"RR 覆盖 {metrics['rr_duration_s']:.2f} 秒；"
                             f"频域连续有效时长 {frequency['supported_duration_s']:.2f} 秒。")
        if frequency["status"] == "insufficient_duration":
            notice = "LF/HF 暂不计算：连续有效 RR 不足 5 分钟。请记录稍长于 5 分钟，补足预热和首尾时间。"
        elif frequency["status"] == "rr_review_required":
            notice = f"LF/HF 暂不计算：{self.current['quality']['suspect_rr_count']} 个 RR 需要复核；未自动删除或补点。"
        else:
            notice = "已计算 LF/HF；结果是 RR 估计，仍需结合峰标记和信号质量核验。"
        self.notice_var.set(notice)
        self.draw()

    def draw_chart(self, canvas, points, xmax, ylabel, bands=False):
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 100 or height < 100:
            return
        if not points:
            canvas.create_text(width/2, height/2, text="暂无可绘制的数据", fill="#aebcd0")
            return
        left, right, top, bottom = 75, width - 24, 30, height - 46
        values = [y for _, y in points]
        low = 0.0 if bands else max(0.0, min(values) - max(10, (max(values)-min(values))*.12))
        high = max(values) * 1.15 if bands else max(values) + max(10, (max(values)-min(values))*.12)
        high = max(high, low + 1)
        xmax = max(xmax, 1e-9)
        if bands:
            for start, end, color, label in ((.04, .15, "#17324f", "LF"), (.15, .4, "#123e3a", "HF")):
                x0, x1 = left + start/xmax*(right-left), left + end/xmax*(right-left)
                canvas.create_rectangle(x0, top, x1, bottom, fill=color, outline="")
                canvas.create_text((x0+x1)/2, top-12, text=label, fill="#afcfda")
        for i in range(5):
            y = top + i/4*(bottom-top)
            canvas.create_line(left, y, right, y, fill="#293a4f")
            canvas.create_text(left-9, y, text=f"{high-i/4*(high-low):.1f}", anchor="e", fill="#aebcd0")
        for i in range(6):
            x = left + i/5*(right-left)
            canvas.create_line(x, top, x, bottom, fill="#293a4f")
            canvas.create_text(x, bottom+15, text=f"{i/5*xmax:.2f}" if bands else f"{i/5*xmax:.0f}", fill="#aebcd0")
        canvas.create_text(left, 13, text=ylabel, anchor="w", fill="#c2cedc")
        canvas.create_text((left+right)/2, height-10, text="频率 / Hz" if bands else "从本段首个 R 峰起 / 秒", fill="#aebcd0")
        coordinates = []
        for stamp, value in points:
            coordinates.extend((left+stamp/xmax*(right-left), bottom-(value-low)/(high-low)*(bottom-top)))
        if len(coordinates) >= 4:
            canvas.create_line(*coordinates, fill="#56dfbb", width=1.8, tags=("data_curve",))

    def draw(self):
        if self.current is None:
            return
        intervals = self.current.get("rr_intervals_ms", [])
        points, stamp = [], 0.0
        for interval in intervals:
            stamp += interval / 1000
            points.append((stamp, interval))
        self.draw_chart(self.rr_canvas, points, stamp, "RR / ms")
        frequency = self.current["frequency"]
        if frequency["status"] == "available_rr_estimate":
            points = [(f, p) for f, p in zip(frequency["frequencies_hz"], frequency["psd_ms2_per_hz"]) if f <= .5]
            self.draw_chart(self.psd_canvas, points, .5, "功率谱密度 / ms²/Hz", bands=True)
        else:
            self.psd_canvas.delete("all")
            self.psd_canvas.create_text(self.psd_canvas.winfo_width()/2, self.psd_canvas.winfo_height()/2,
                                       text=self.notice_var.get(), width=700, fill="#b8c7da", justify="center")

    def save_as(self):
        filename = filedialog.asksaveasfilename(parent=self, title="另存 HRV 报告", defaultextension=".json",
                                               initialfile="hrv_report.json", filetypes=[("JSON", "*.json")])
        if not filename:
            return
        try:
            save_report_new(self.report, Path(filename))
            self.path_var.set(f"报告：{filename}")
        except OSError as error:
            messagebox.showerror("报告未保存", str(error), parent=self)
