# tab_capture.py
from PyQt5 import QtWidgets, QtCore
import os
import pcapy  # pcapy-ng

SETTINGS_FILE = "capture_settings.ini"
WATCHDOG_TYPE_ORDER = (
    "IDLE",
    "TRIG_ONLY",
    "START_EVENT",
    "CAPTURE_PRE",
    "SEND_HEAD",
    "FLUSH_BUF",
    "STREAM_LIVE",
    "SEND_TRAIL",
    "ABORT_MARK",
    "ABORT_TRAIL",
    "ABORT_CLEAN",
    "legacy/unknown",
)


class tab_capture(QtCore.QObject):
    def __init__(self, parent_widget, backend=None, *, initial_rt_file: str = "", rt_apply_callback=None):
        super().__init__(parent_widget)
        self.parent = parent_widget
        self.backend = backend
        self._initial_rt_file = str(initial_rt_file or "")
        self._rt_apply_callback = rt_apply_callback


        settings_path = os.path.join(os.getcwd(), SETTINGS_FILE)
        self.settings = QtCore.QSettings(settings_path, QtCore.QSettings.IniFormat)

        self._build_ui()
        self._load_settings()
        self.refresh_devices()
        self._watchdog_detail_dialog = None
        self._watchdog_detail_table = None

        if self.backend is not None:
            self.backend.stats.connect(self.update_stats)
            self.backend.analysis_1hz.connect(self._on_decode_1hz)
            if hasattr(self.backend, "run_finished"):
                self.backend.run_finished.connect(self._on_run_finished)

        self._last_decode = None  # last DecodeSnapshot

    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self.parent)

        # device row
        dev_row = QtWidgets.QHBoxLayout()
        layout.addLayout(dev_row)

        dev_row.addWidget(QtWidgets.QLabel("Device:"))

        self.combo_iface = QtWidgets.QComboBox()
        self.combo_iface.setMinimumWidth(260)
        self.combo_iface.currentIndexChanged.connect(self._on_device_changed)
        dev_row.addWidget(self.combo_iface, 1)

        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_devices)
        dev_row.addWidget(self.btn_refresh)

        # form
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.edit_filter = QtWidgets.QLineEdit()
        self.edit_filter.setText("ether src ff:ff:ff:c7:05:01")
        self.edit_filter.editingFinished.connect(self._save_settings)
        form.addRow("BPF filter:", self.edit_filter)

        self.edit_outdir = QtWidgets.QLineEdit()
        self.edit_outdir.setText(os.getcwd())
        self.edit_outdir.editingFinished.connect(self._save_settings)
        outdir_row = QtWidgets.QHBoxLayout()
        outdir_row.addWidget(self.edit_outdir, 1)
        self.btn_browse_outdir = QtWidgets.QPushButton("Browse...")
        self.btn_browse_outdir.clicked.connect(self._browse_output_dir)
        outdir_row.addWidget(self.btn_browse_outdir)
        form.addRow("Output dir:", outdir_row)

        dat_row = QtWidgets.QHBoxLayout()
        self.edit_datfile = QtWidgets.QLineEdit()
        self.edit_datfile.setPlaceholderText("Select a local .dat file to replay")
        self.edit_datfile.editingFinished.connect(self._save_settings)
        dat_row.addWidget(self.edit_datfile, 1)
        self.btn_browse_dat = QtWidgets.QPushButton("Browse...")
        self.btn_browse_dat.clicked.connect(self._browse_dat_file)
        dat_row.addWidget(self.btn_browse_dat)
        form.addRow("Offline DAT:", dat_row)

        rt_row = QtWidgets.QHBoxLayout()
        self.edit_rtfile = QtWidgets.QLineEdit()
        self.edit_rtfile.setPlaceholderText("Select RT coefficient .txt file")
        self.edit_rtfile.editingFinished.connect(self._save_settings)
        rt_row.addWidget(self.edit_rtfile, 1)
        self.btn_browse_rt = QtWidgets.QPushButton("Browse...")
        self.btn_browse_rt.clicked.connect(self._browse_rt_file)
        rt_row.addWidget(self.btn_browse_rt)
        form.addRow("RT file:", rt_row)

        self.lab_current_dat = QtWidgets.QLabel("-")
        self.lab_current_dat.setWordWrap(True)
        form.addRow("Current DAT:", self.lab_current_dat)

        run_row = QtWidgets.QHBoxLayout()
        self.spin_run = QtWidgets.QSpinBox()
        self.spin_run.setRange(1, 99999999)
        self.spin_run.setValue(1)
        self.spin_run.valueChanged.connect(self._on_run_changed)
        run_row.addWidget(self.spin_run)
        run_row.addStretch(1)
        form.addRow("Next run #:", run_row)

        # buttons
        btns = QtWidgets.QHBoxLayout()
        layout.addLayout(btns)

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.clicked.connect(self.start)
        btns.addWidget(self.btn_start)

        self.btn_process = QtWidgets.QPushButton("Process DAT")
        self.btn_process.clicked.connect(self.process_dat)
        btns.addWidget(self.btn_process)

        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        btns.addWidget(self.btn_stop)

        # capture stats
        stats = QtWidgets.QGridLayout()
        layout.addLayout(stats)

        self.lab_total = QtWidgets.QLabel("0")
        self.lab_lost = QtWidgets.QLabel("0")
        self.lab_bytes = QtWidgets.QLabel("0")
        self.lab_file = QtWidgets.QLabel("-")

        stats.addWidget(QtWidgets.QLabel("Total packets:"), 0, 0)
        stats.addWidget(self.lab_total, 0, 1)
        stats.addWidget(QtWidgets.QLabel("Lost (total):"), 1, 0)
        stats.addWidget(self.lab_lost, 1, 1)
        stats.addWidget(QtWidgets.QLabel("Buffered bytes (total):"), 2, 0)
        stats.addWidget(self.lab_bytes, 2, 1)
        stats.addWidget(QtWidgets.QLabel("Current file:"), 3, 0)
        stats.addWidget(self.lab_file, 3, 1)

        # decode stats
        dec = QtWidgets.QGridLayout()
        layout.addLayout(dec)

        self.lab_hdr = QtWidgets.QLabel("0")
        self.lab_fifoA = QtWidgets.QLabel("0")
        self.lab_fifoB = QtWidgets.QLabel("0")
        self.lab_trl = QtWidgets.QLabel("0")
        self.lab_trg = QtWidgets.QLabel("0")
        self.lab_hit = QtWidgets.QLabel("0")
        self.lab_evbuf = QtWidgets.QLabel("0")

        self.lab_err_eid = QtWidgets.QLabel("0")
        self.lab_err_hit = QtWidgets.QLabel("0")
        self.lab_err_mtrl = QtWidgets.QLabel("0")
        self.lab_err_mhdr = QtWidgets.QLabel("0")
        self.lab_abort_burst = QtWidgets.QLabel("0")
        self.lab_abort_watchdog = QtWidgets.QLabel("0")
        self.btn_abort_watchdog_details = QtWidgets.QPushButton("Details...")
        self.btn_abort_watchdog_details.clicked.connect(self._show_watchdog_details)

        counts_form = QtWidgets.QFormLayout()
        counts_form.addRow("Headers:", self.lab_hdr)
        counts_form.addRow("Events: fifoA:", self.lab_fifoA)
        counts_form.addRow("Events: fifoB:", self.lab_fifoB)
        counts_form.addRow("Trailers:", self.lab_trl)
        counts_form.addRow("Hits:", self.lab_hit)
        counts_form.addRow("Triggers:", self.lab_trg)
        counts_form.addRow("Events buffered:", self.lab_evbuf)

        errors_form = QtWidgets.QFormLayout()
        errors_form.addRow("Err: event_id mismatch:", self.lab_err_eid)
        errors_form.addRow("Err: hit_count mismatch:", self.lab_err_hit)
        errors_form.addRow("Err: missing trailer:", self.lab_err_mtrl)
        errors_form.addRow("Err: missing header:", self.lab_err_mhdr)

        aborts_form = QtWidgets.QFormLayout()
        aborts_form.addRow("Abort: burst:", self.lab_abort_burst)
        watchdog_row = QtWidgets.QHBoxLayout()
        watchdog_row.setContentsMargins(0, 0, 0, 0)
        watchdog_row.addWidget(self.lab_abort_watchdog)
        watchdog_row.addWidget(self.btn_abort_watchdog_details)
        watchdog_row.addStretch(1)
        aborts_form.addRow("Abort: watchdog:", watchdog_row)

        dec.addWidget(QtWidgets.QLabel("<b>Counts</b>"), 0, 0)
        dec.addWidget(QtWidgets.QLabel("<b>Err / Mismatch</b>"), 0, 1)
        dec.addWidget(QtWidgets.QLabel("<b>Aborts</b>"), 0, 2)
        dec.addLayout(counts_form, 1, 0)
        dec.addLayout(errors_form, 1, 1)
        dec.addLayout(aborts_form, 1, 2)
        dec.setColumnStretch(0, 1)
        dec.setColumnStretch(1, 1)
        dec.setColumnStretch(2, 2)

        layout.addStretch(1)

    # ------------------------------------------------------------------
    # settings

    def _load_settings(self):
        self._last_device = self.settings.value("last/last_device", "", type=str)
        last_filter = self.settings.value("last/last_filter", "", type=str)
        last_outdir = self.settings.value("last/out_dir", "", type=str)
        last_dat = self.settings.value("last/dat_file", "", type=str)
        last_rt = self.settings.value("last/rt_file", self._initial_rt_file, type=str)

        if last_filter:
            self.edit_filter.setText(last_filter)

        last_outdir = os.path.expanduser((last_outdir or "").strip())
        if last_outdir and os.path.isdir(last_outdir):
            self.edit_outdir.setText(last_outdir)
        else:
            self.edit_outdir.setText(os.getcwd())

        last_dat = os.path.expanduser((last_dat or "").strip())
        if last_dat and os.path.isfile(last_dat):
            self.edit_datfile.setText(last_dat)
        else:
            self.edit_datfile.clear()

        last_rt = os.path.expanduser((last_rt or "").strip())
        initial_rt = os.path.expanduser((self._initial_rt_file or "").strip())
        if last_rt and os.path.isfile(last_rt):
            self.edit_rtfile.setText(last_rt)
        elif initial_rt and os.path.isfile(initial_rt):
            self.edit_rtfile.setText(initial_rt)
        else:
            self.edit_rtfile.clear()

        next_run = self.settings.value("run/next_run", 1, type=int)
        if next_run < 1:
            next_run = 1
        self.spin_run.setValue(int(next_run))

    def _save_settings(self):
        self.settings.setValue("last/last_device", self.current_device())
        self.settings.setValue("last/last_filter", self.edit_filter.text().strip())
        self.settings.setValue("last/out_dir", self.edit_outdir.text().strip())
        self.settings.setValue("last/dat_file", self.edit_datfile.text().strip())
        self.settings.setValue("last/rt_file", self.edit_rtfile.text().strip())
        self.settings.setValue("run/next_run", int(self.spin_run.value()))
        self.settings.sync()

    def _on_run_changed(self, _val: int):
        self._save_settings()

    # ------------------------------------------------------------------
    # devices

    def refresh_devices(self):
        self.combo_iface.blockSignals(True)
        self.combo_iface.clear()

        try:
            devs = pcapy.findalldevs()
        except Exception as e:
            self.combo_iface.addItem("ERROR listing devices")
            self.combo_iface.blockSignals(False)
            print(f"[ERROR] pcapy.findalldevs() failed: {repr(e)}")
            return

        for d in devs:
            self.combo_iface.addItem(d)

        self.combo_iface.blockSignals(False)

        if not devs:
            return

        if self._last_device and self._last_device in devs:
            self.combo_iface.setCurrentIndex(devs.index(self._last_device))
        else:
            pick = next((d for d in devs if d != "lo"), devs[0])
            self.combo_iface.setCurrentIndex(devs.index(pick))

    def current_device(self) -> str:
        return self.combo_iface.currentText().strip()

    def _on_device_changed(self, _idx: int):
        dev = self.current_device()
        if dev and "ERROR" not in dev:
            self._save_settings()

    def _dialog_start_dir(self, path_text: str = "") -> str:
        path = os.path.expanduser((path_text or "").strip())
        if path and os.path.isdir(path):
            return path
        if path:
            parent = os.path.dirname(path)
            if parent and os.path.isdir(parent):
                return parent
        return os.getcwd()

    def _browse_output_dir(self):
        start_dir = self._dialog_start_dir(self.edit_outdir.text())
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self.parent,
            "Select output directory",
            start_dir,
        )
        if not path:
            return
        self.edit_outdir.setText(path)
        self._save_settings()

    def _browse_dat_file(self):
        start_dir = self._dialog_start_dir(self.edit_datfile.text() or self.edit_outdir.text())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent,
            "Select DAT file",
            start_dir,
            "DAT files (*.dat);;All files (*)",
        )
        if not path:
            return
        self.edit_datfile.setText(path)
        self._save_settings()

    def _browse_rt_file(self):
        start_dir = self._dialog_start_dir(self.edit_rtfile.text() or self.edit_outdir.text())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent,
            "Select RT file",
            start_dir,
            "TXT files (*.txt);;All files (*)",
        )
        if not path:
            return
        self.edit_rtfile.setText(path)
        self._save_settings()

    def _set_current_dat_name(self, filename: str):
        name = os.path.basename((filename or "").strip()) or "-"
        self.lab_current_dat.setText(name)
        self.lab_file.setText(name)

    def _apply_rt_selection(self) -> bool:
        if self._rt_apply_callback is None:
            return True
        rt_path = self.edit_rtfile.text().strip()
        ok = bool(self._rt_apply_callback(rt_path))
        if not ok:
            print("[ERROR] Failed to apply RT file selection.")
        return ok

    # ------------------------------------------------------------------
    # run control

    def _allocate_run_number(self) -> int:
        run = int(self.spin_run.value())
        self.spin_run.setValue(run + 1)
        return run

    def start(self):
        if self.backend is None:
            print("[ERROR] Backend is not initialized.")
            return

        dev = self.current_device()
        bpf = self.edit_filter.text().strip()
        out_dir = self.edit_outdir.text().strip() or os.getcwd()

        if not dev or "ERROR" in dev:
            print("[ERROR] No valid device selected.")
            return

        if not self._apply_rt_selection():
            return

        self._save_settings()

        run = self._allocate_run_number()
        out_path = self.backend.make_out_path(out_dir, run)

        print(f"Starting capture on {dev}")
        print(f"Filter: {bpf}")
        print(f"Output: {out_path}")

        self.backend.start_capture(dev, bpf, out_path)

        self._set_current_dat_name(out_path)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop(self):
        if self.backend:
            self.backend.stop_capture()

        self.btn_start.setEnabled(True)
        self.btn_process.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def process_dat(self):
        if self.backend is None:
            print("[ERROR] Backend is not initialized.")
            return

        dat_path = self.edit_datfile.text().strip()
        if not dat_path:
            print("[ERROR] Please select a .dat file first.")
            return
        if not os.path.isfile(dat_path):
            print(f"[ERROR] DAT file not found: {dat_path}")
            return

        if not self._apply_rt_selection():
            return

        self._save_settings()
        print(f"Processing DAT: {dat_path}")
        self.backend.start_replay(dat_path)

        self._set_current_dat_name(dat_path)
        self.btn_start.setEnabled(False)
        self.btn_process.setEnabled(False)
        self.btn_stop.setEnabled(True)

    # ------------------------------------------------------------------
    # slots

    @QtCore.pyqtSlot(int, int, int, str)
    def update_stats(self, total_packets, lost_total, bytes_total, filename):
        self.lab_total.setText(str(total_packets))
        self.lab_lost.setText(str(lost_total))
        self.lab_bytes.setText(str(bytes_total))
        self._set_current_dat_name(filename)

    @staticmethod
    def _ordered_watchdog_types(type_names):
        known = set(WATCHDOG_TYPE_ORDER)
        present = set(type_names)
        ordered = [name for name in WATCHDOG_TYPE_ORDER if name in present]
        extra = sorted(name for name in present if name not in known)
        return ordered + extra

    def _current_watchdog_type_counts(self):
        if self._last_decode is None:
            return {}
        return {str(name): int(count) for name, count in getattr(self._last_decode, "abort_watchdog_types", ())}

    def _ensure_watchdog_detail_dialog(self):
        if self._watchdog_detail_dialog is not None:
            return

        dialog = QtWidgets.QDialog(self.parent)
        dialog.setWindowTitle("Watchdog Abort Details")
        dialog.resize(360, 320)
        layout = QtWidgets.QVBoxLayout(dialog)

        table = QtWidgets.QTableWidget(0, 2, dialog)
        table.setHorizontalHeaderLabels(["Type", "Count"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setFocusPolicy(QtCore.Qt.NoFocus)
        table.setAlternatingRowColors(True)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        layout.addWidget(table)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close, parent=dialog)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        self._watchdog_detail_dialog = dialog
        self._watchdog_detail_table = table

    def _refresh_watchdog_detail_dialog(self):
        if self._watchdog_detail_table is None:
            return

        counts = self._current_watchdog_type_counts()
        ordered_types = self._ordered_watchdog_types(tuple(WATCHDOG_TYPE_ORDER) + tuple(counts.keys()))
        self._watchdog_detail_table.setRowCount(len(ordered_types))

        for row, watchdog_type in enumerate(ordered_types):
            name_item = QtWidgets.QTableWidgetItem(str(watchdog_type))
            count_item = QtWidgets.QTableWidgetItem(str(int(counts.get(watchdog_type, 0))))
            count_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self._watchdog_detail_table.setItem(row, 0, name_item)
            self._watchdog_detail_table.setItem(row, 1, count_item)

    def _show_watchdog_details(self):
        self._ensure_watchdog_detail_dialog()
        self._refresh_watchdog_detail_dialog()
        self._watchdog_detail_dialog.show()
        self._watchdog_detail_dialog.raise_()
        self._watchdog_detail_dialog.activateWindow()

    @QtCore.pyqtSlot(object)
    def _on_decode_1hz(self, snap):
        self._last_decode = snap

        self.lab_hdr.setText(str(int(snap.headers)))
        self.lab_fifoA.setText(str(int(getattr(snap, "events_fifoA", 0))))
        self.lab_fifoB.setText(str(int(getattr(snap, "events_fifoB", 0))))
        self.lab_trl.setText(str(int(snap.trailers)))
        self.lab_trg.setText(str(int(snap.triggers)))
        self.lab_hit.setText(str(int(snap.hits_total)))
        self.lab_evbuf.setText(str(int(snap.events_buffered)))
        self.lab_abort_burst.setText(str(int(getattr(snap, "abort_burst", 0))))
        self.lab_abort_watchdog.setText(str(int(getattr(snap, "abort_watchdog", 0))))
        if self._watchdog_detail_dialog is not None and self._watchdog_detail_dialog.isVisible():
            self._refresh_watchdog_detail_dialog()

        self.lab_err_eid.setText(str(int(snap.err_event_id)))
        self.lab_err_hit.setText(str(int(snap.err_hit_count)))
        self.lab_err_mtrl.setText(str(int(snap.err_missing_trailer)))
        self.lab_err_mhdr.setText(str(int(snap.err_missing_header)))

    @QtCore.pyqtSlot()
    def _on_run_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_process.setEnabled(True)
        self.btn_stop.setEnabled(False)
