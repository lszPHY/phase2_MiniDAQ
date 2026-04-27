# geometry_dialog.py
from __future__ import annotations

import os
from typing import List, Tuple, Optional

from PyQt5 import QtWidgets

from geometry import Geometry


class GeometryConfigDialog(QtWidgets.QDialog):
    """
    One dialog to:
      - edit ALL geometry settings
      - assign TDC slots (ML0/ML1: slot -> (tdc_id, ncol))
      - Load/Save to a single geometry.txt via file browser

    Slot packing rule:
      slot i starts at sum(ncol of slots 0..i-1)
      slot i covers [start, start + ncol - 1]
    """

    def __init__(
        self,
        parent=None,
        geo: Optional[Geometry] = None,
        slots_per_ml: int = 10,
        ml0_slots: Optional[List[Tuple[int, int]]] = None,
        ml1_slots: Optional[List[Tuple[int, int]]] = None,
        default_path: str = "geometry.txt",
    ):
        super().__init__(parent)
        self.setWindowTitle("Geometry + TDC Slot Assignment")
        self.setModal(True)

        self._geo_in = geo if geo is not None else Geometry()
        self._slots_per_ml = int(slots_per_ml)
        self._default_path = str(default_path)

        self._ml0 = list(ml0_slots or [])
        self._ml1 = list(ml1_slots or [])

        self._build_ui()
        self._load_into_widgets(self._geo_in, self._slots_per_ml, self._ml0, self._ml1)

        self.resize(940, 680)

    # ---------------- UI ----------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # IO row
        io = QtWidgets.QHBoxLayout()
        layout.addLayout(io)

        self.btn_load = QtWidgets.QPushButton("Load...")
        self.btn_save = QtWidgets.QPushButton("Save...")
        io.addWidget(self.btn_load)
        io.addWidget(self.btn_save)

        self.lab_io = QtWidgets.QLabel("")
        io.addWidget(self.lab_io, 1)

        self.btn_load.clicked.connect(self._on_load)
        self.btn_save.clicked.connect(self._on_save)

        # Tabs: Settings, ML0, ML1
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        self._tab_settings = QtWidgets.QWidget()
        self._tab_ml0 = QtWidgets.QWidget()
        self._tab_ml1 = QtWidgets.QWidget()
        self.tabs.addTab(self._tab_settings, "Geometry settings")
        self.tabs.addTab(self._tab_ml0, "ML0 slots")
        self.tabs.addTab(self._tab_ml1, "ML1 slots")

        self._build_settings_tab(self._tab_settings)
        self._rows0 = self._build_slots_tab(self._tab_ml0)
        self._rows1 = self._build_slots_tab(self._tab_ml1)

        # bottom OK/Cancel
        bottom = QtWidgets.QHBoxLayout()
        layout.addLayout(bottom)
        self.lab_warn = QtWidgets.QLabel("")
        bottom.addWidget(self.lab_warn, 1)

        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_ok = QtWidgets.QPushButton("OK")
        bottom.addWidget(self.btn_cancel)
        bottom.addWidget(self.btn_ok)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self._on_ok)

        # live rebuild if slots_per_ml changes
        self.sb_slots_per_ml.valueChanged.connect(self._on_slots_per_ml_changed)

    def _build_settings_tab(self, w: QtWidgets.QWidget):
        form = QtWidgets.QFormLayout(w)

        # ---- general settings ----
        self.cb_chamber = QtWidgets.QComboBox()
        self.cb_chamber.addItems(["A", "C"])
        form.addRow("chamberType", self.cb_chamber)

        self.sb_flip = QtWidgets.QSpinBox()
        self.sb_flip.setRange(0, 1)
        form.addRow("flipTDCs", self.sb_flip)

        self.sb_tdcColByTubeNo = QtWidgets.QSpinBox()
        self.sb_tdcColByTubeNo.setRange(0, 1)
        form.addRow("tdcColByTubeNo", self.sb_tdcColByTubeNo)

        # ---- integer dims ----
        self.sb_MAX_TDC = QtWidgets.QSpinBox()
        self.sb_MAX_TDC.setRange(1, 512)
        self.sb_MAX_TDC_CHANNEL = QtWidgets.QSpinBox()
        self.sb_MAX_TDC_CHANNEL.setRange(1, 64)
        self.sb_MAX_TUBE_LAYER = QtWidgets.QSpinBox()
        self.sb_MAX_TUBE_LAYER.setRange(1, 64)
        self.sb_MAX_TUBE_COLUMN = QtWidgets.QSpinBox()
        self.sb_MAX_TUBE_COLUMN.setRange(1, 512)
        self.sb_MAX_TDC_LAYER = QtWidgets.QSpinBox()
        self.sb_MAX_TDC_LAYER.setRange(1, 16)

        form.addRow("MAX_TDC", self.sb_MAX_TDC)
        form.addRow("MAX_TDC_CHANNEL", self.sb_MAX_TDC_CHANNEL)
        form.addRow("MAX_TUBE_LAYER", self.sb_MAX_TUBE_LAYER)
        form.addRow("MAX_TUBE_COLUMN", self.sb_MAX_TUBE_COLUMN)
        form.addRow("MAX_TDC_LAYER", self.sb_MAX_TDC_LAYER)

        # ---- floats ----
        def _dspin(minv=-1e9, maxv=1e9, step=0.1, decimals=6):
            ds = QtWidgets.QDoubleSpinBox()
            ds.setRange(minv, maxv)
            ds.setDecimals(decimals)
            ds.setSingleStep(step)
            return ds

        self.ds_ML_distance = _dspin(step=0.1, decimals=6)
        self.ds_tube_length = _dspin(step=0.001, decimals=6)
        self.ds_layer_distance = _dspin(step=0.001, decimals=9)
        self.ds_column_distance = _dspin(step=0.001, decimals=6)
        self.ds_radius = _dspin(step=0.1, decimals=6)
        self.ds_min_drift = _dspin(step=0.01, decimals=6)
        self.ds_max_drift = _dspin(step=0.01, decimals=6)

        form.addRow("ML_distance", self.ds_ML_distance)
        form.addRow("tube_length", self.ds_tube_length)
        form.addRow("layer_distance", self.ds_layer_distance)
        form.addRow("column_distance", self.ds_column_distance)
        form.addRow("radius", self.ds_radius)
        form.addRow("min_drift_dist", self.ds_min_drift)
        form.addRow("max_drift_dist", self.ds_max_drift)

        # ---- slots per ML ----
        self.sb_slots_per_ml = QtWidgets.QSpinBox()
        self.sb_slots_per_ml.setRange(1, 128)
        form.addRow("slots_per_ml", self.sb_slots_per_ml)

        note = QtWidgets.QLabel(
            "Packed slots: slot i starts at sum(ncol of previous slots). "
            "ncol is number of valid columns covered by that slot."
        )
        note.setWordWrap(True)
        form.addRow(note)

    def _build_slots_tab(self, w: QtWidgets.QWidget):
        v = QtWidgets.QVBoxLayout(w)

        grid = QtWidgets.QGridLayout()
        v.addLayout(grid)

        grid.addWidget(QtWidgets.QLabel("Slot"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("TDC ID"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("ncol (#cols)"), 0, 2)

        rows: List[Tuple[QtWidgets.QComboBox, QtWidgets.QSpinBox]] = []

        # tools
        tools = QtWidgets.QHBoxLayout()
        v.addLayout(tools)

        btn_all6 = QtWidgets.QPushButton("Set all ncol=6")
        btn_clamp = QtWidgets.QPushButton("Clamp by MAX_TUBE_COLUMN")
        btn_clear = QtWidgets.QPushButton("Clear (TDC=--)")
        tools.addWidget(btn_all6)
        tools.addWidget(btn_clamp)
        tools.addWidget(btn_clear)
        tools.addStretch(1)

        def _do_all6():
            for _, sb in rows:
                sb.setValue(6)

        def _do_clear():
            for cb, _ in rows:
                cb.setCurrentIndex(0)

        def _do_clamp():
            max_cols = int(self.sb_MAX_TUBE_COLUMN.value())
            acc = 0
            for _, sb in rows:
                ncol = max(0, int(sb.value()))
                if acc >= max_cols:
                    sb.setValue(0)
                    continue
                ncol2 = min(ncol, max_cols - acc)
                sb.setValue(ncol2)
                acc += ncol2

        btn_all6.clicked.connect(_do_all6)
        btn_clear.clicked.connect(_do_clear)
        btn_clamp.clicked.connect(_do_clamp)

        return rows

    # ---------------- value plumbing ----------------

    def _rebuild_slots_rows(
        self,
        tab: QtWidgets.QWidget,
        rows: List[Tuple[QtWidgets.QComboBox, QtWidgets.QSpinBox]],
        init_vals: List[Tuple[int, int]],
    ):
        grid = tab.layout().itemAt(0).layout()
        assert isinstance(grid, QtWidgets.QGridLayout)

        # delete previous widgets (except header row)
        for r in range(1, grid.rowCount() + 1):
            for c in range(0, 3):
                it = grid.itemAtPosition(r, c)
                if it and it.widget():
                    it.widget().deleteLater()

        rows.clear()

        slots = int(self.sb_slots_per_ml.value())
        n_tdcs = int(self.sb_MAX_TDC.value())

        for i in range(slots):
            lab = QtWidgets.QLabel(f"{i:02d}")

            cb = QtWidgets.QComboBox()
            cb.addItem("--", -1)
            for tdc in range(n_tdcs):
                cb.addItem(f"{tdc:02d}", tdc)

            sb = QtWidgets.QSpinBox()
            sb.setRange(0, 512)
            sb.setSingleStep(1)

            tdc_id, ncol = init_vals[i] if i < len(init_vals) else (-1, 6)
            cb.setCurrentIndex(0 if int(tdc_id) < 0 else int(tdc_id) + 1)
            sb.setValue(max(0, int(ncol)))

            grid.addWidget(lab, i + 1, 0)
            grid.addWidget(cb, i + 1, 1)
            grid.addWidget(sb, i + 1, 2)

            rows.append((cb, sb))

    def _load_into_widgets(self, geo: Geometry, slots_per_ml: int, ml0: List[Tuple[int, int]], ml1: List[Tuple[int, int]]):
        self.cb_chamber.setCurrentText(str(geo.chamberType))
        self.sb_flip.setValue(int(geo.flipTDCs))
        self.sb_tdcColByTubeNo.setValue(int(geo.tdcColByTubeNo))

        self.sb_MAX_TDC.setValue(int(geo.MAX_TDC))
        self.sb_MAX_TDC_CHANNEL.setValue(int(geo.MAX_TDC_CHANNEL))
        self.sb_MAX_TUBE_LAYER.setValue(int(geo.MAX_TUBE_LAYER))
        self.sb_MAX_TUBE_COLUMN.setValue(int(geo.MAX_TUBE_COLUMN))
        self.sb_MAX_TDC_LAYER.setValue(int(geo.MAX_TDC_LAYER))

        self.ds_ML_distance.setValue(float(geo.ML_distance))
        self.ds_tube_length.setValue(float(geo.tube_length))
        self.ds_layer_distance.setValue(float(geo.layer_distance))
        self.ds_column_distance.setValue(float(geo.column_distance))
        self.ds_radius.setValue(float(geo.radius))
        self.ds_min_drift.setValue(float(geo.min_drift_dist))
        self.ds_max_drift.setValue(float(geo.max_drift_dist))

        self.sb_slots_per_ml.blockSignals(True)
        self.sb_slots_per_ml.setValue(int(slots_per_ml))
        self.sb_slots_per_ml.blockSignals(False)

        self._rebuild_slots_rows(self._tab_ml0, self._rows0, ml0)
        self._rebuild_slots_rows(self._tab_ml1, self._rows1, ml1)

    def _read_geo_from_widgets(self) -> Geometry:
        # NOTE: we do NOT pass geo_file here; dialog edits pure geometry values.
        return Geometry(
            chamberType=str(self.cb_chamber.currentText()),
            flipTDCs=int(self.sb_flip.value()),
            tdcColByTubeNo=int(self.sb_tdcColByTubeNo.value()),
            MAX_TDC=int(self.sb_MAX_TDC.value()),
            MAX_TDC_CHANNEL=int(self.sb_MAX_TDC_CHANNEL.value()),
            MAX_TUBE_LAYER=int(self.sb_MAX_TUBE_LAYER.value()),
            MAX_TUBE_COLUMN=int(self.sb_MAX_TUBE_COLUMN.value()),
            MAX_TDC_LAYER=int(self.sb_MAX_TDC_LAYER.value()),
            ML_distance=float(self.ds_ML_distance.value()),
            tube_length=float(self.ds_tube_length.value()),
            layer_distance=float(self.ds_layer_distance.value()),
            column_distance=float(self.ds_column_distance.value()),
            radius=float(self.ds_radius.value()),
            min_drift_dist=float(self.ds_min_drift.value()),
            max_drift_dist=float(self.ds_max_drift.value()),
        )

    def _read_slots_from_rows(self, rows: List[Tuple[QtWidgets.QComboBox, QtWidgets.QSpinBox]]) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for cb, sb in rows:
            out.append((int(cb.currentData()), max(0, int(sb.value()))))
        return out

    def _on_slots_per_ml_changed(self, _):
        ml0 = self._read_slots_from_rows(self._rows0) if self._rows0 else []
        ml1 = self._read_slots_from_rows(self._rows1) if self._rows1 else []
        self._rebuild_slots_rows(self._tab_ml0, self._rows0, ml0)
        self._rebuild_slots_rows(self._tab_ml1, self._rows1, ml1)

    # ---------------- button actions ----------------

    def _on_save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save geometry", self._default_path, "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        try:
            geo = self._read_geo_from_widgets()
            slots = int(self.sb_slots_per_ml.value())
            ml0 = self._read_slots_from_rows(self._rows0)
            ml1 = self._read_slots_from_rows(self._rows1)

            # Store assignment inside the Geometry and save using Geometry.save()
            geo.set_assignment(slots, ml0, ml1, apply_map=True)
            geo.save(path)

            self.lab_io.setText(f"Saved: {path}")
        except Exception as e:
            self.lab_io.setText(f"Save failed: {type(e).__name__}: {e}")

    def _on_load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load geometry", "", "Text files (*.txt);;All files (*)"
        )
        if not path:
            return

        try:
            # Load directly from Geometry
            geo = Geometry.load(path, apply_map=True)

            # Pull assignment from the loaded Geometry
            slots = int(geo.slots_per_ml)
            ml0 = list(geo.ml0)
            ml1 = list(geo.ml1)

            # apply loaded slots_per_ml before rebuild
            self._load_into_widgets(geo, slots, ml0, ml1)
            self.tabs.setCurrentIndex(0)

            self.lab_io.setText(
                f"Loaded: {os.path.basename(path)} | "
                f"MAX_TDC={geo.MAX_TDC} MAX_TUBE_COLUMN={geo.MAX_TUBE_COLUMN} slots_per_ml={slots}"
            )
        except Exception as e:
            self.lab_io.setText(f"Load failed: {type(e).__name__}: {e}")

    def _on_ok(self):
        geo = self._read_geo_from_widgets()
        slots = int(self.sb_slots_per_ml.value())
        ml0 = self._read_slots_from_rows(self._rows0)[:slots]
        ml1 = self._read_slots_from_rows(self._rows1)[:slots]

        ml0_ids = [tdc for tdc, ncol in ml0 if tdc >= 0 and ncol > 0]
        ml1_ids = [tdc for tdc, ncol in ml1 if tdc >= 0 and ncol > 0]

        warn: List[str] = []
        if len(set(ml0_ids)) != len(ml0_ids):
            warn.append("ML0 has duplicate TDC IDs")
        if len(set(ml1_ids)) != len(ml1_ids):
            warn.append("ML1 has duplicate TDC IDs")
        if len(set(ml0_ids + ml1_ids)) != len(ml0_ids + ml1_ids):
            warn.append("Duplicates across ML0/ML1")

        tot0 = sum(max(0, int(n)) for _, n in ml0)
        tot1 = sum(max(0, int(n)) for _, n in ml1)
        if tot0 > geo.MAX_TUBE_COLUMN:
            warn.append(f"ML0 sum(ncol)={tot0} exceeds MAX_TUBE_COLUMN={geo.MAX_TUBE_COLUMN}")
        if tot1 > geo.MAX_TUBE_COLUMN:
            warn.append(f"ML1 sum(ncol)={tot1} exceeds MAX_TUBE_COLUMN={geo.MAX_TUBE_COLUMN}")

        self.lab_warn.setText(" | ".join(warn))

        # Commit assignment into the dialog's resulting Geometry object.
        geo.set_assignment(slots, ml0, ml1, apply_map=True)

        self.accept()

    # ---------------- public getters ----------------

    def result_geometry(self) -> Geometry:
        """
        Return a Geometry instance that includes the slot assignment.
        """
        geo = self._read_geo_from_widgets()
        slots = int(self.sb_slots_per_ml.value())
        ml0 = self._read_slots_from_rows(self._rows0)[:slots]
        ml1 = self._read_slots_from_rows(self._rows1)[:slots]
        geo.set_assignment(slots, ml0, ml1, apply_map=True)
        return geo

    def result_slots_per_ml(self) -> int:
        return int(self.sb_slots_per_ml.value())

    def result_ml0_slots(self) -> List[Tuple[int, int]]:
        slots = int(self.sb_slots_per_ml.value())
        vals = self._read_slots_from_rows(self._rows0)
        return vals[:slots]

    def result_ml1_slots(self) -> List[Tuple[int, int]]:
        slots = int(self.sb_slots_per_ml.value())
        vals = self._read_slots_from_rows(self._rows1)
        return vals[:slots]
