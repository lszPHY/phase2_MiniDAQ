# MainWindow.py
from PyQt5 import QtWidgets

from tab_capture import tab_capture
from tab_geometry import tab_geometry

# NEW: import the concrete spectra tabs
from tab_spectra import (
    tab_adc_spectra,
    tab_tdc_spectra,
    tab_adc_channels,
    tab_tdc_channels,
    tab_adc_channel_hits,
    tab_fit_residual_spectra,
    tab_fit_residual_overall,
)


class Ui_MainWindow:
    def __init__(self):
        pass

    def setupUi(self, MainWindow, backend=None, geo=None, rt=None):
        """
        geo is expected to be a list/tuple of Geometry objects: [geo0, geo1, ...]
        Each Geometry MUST have geo.chamber_id set (>=0).
        """
        MainWindow.setWindowTitle("GUI_DAQ")
        self.backend = backend

        if geo is None:
            geos = []
        elif isinstance(geo, (list, tuple)):
            geos = list(geo)
        else:
            # allow single Geometry
            geos = [geo]

        if not geos:
            raise ValueError("MainWindow.setupUi expects geo=[Geometry,...] (non-empty)")

        # validate chamber_id and build a stable ordering (by chamber_id)
        for g in geos:
            if g is None:
                raise ValueError("MainWindow.setupUi got geo list containing None")
            cid = int(getattr(g, "chamber_id", -1))
            if cid < 0:
                raise ValueError("Each Geometry must have geo.chamber_id >= 0")

        geos.sort(key=lambda g: int(getattr(g, "chamber_id", -1)))
        self.geos = geos  # keep reference
        self.rt = rt

        self.centralWidget = QtWidgets.QWidget()
        MainWindow.setCentralWidget(self.centralWidget)
        mainLayout = QtWidgets.QVBoxLayout(self.centralWidget)

        self.tabWidget = QtWidgets.QTabWidget()
        self.tabWidget.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        mainLayout.addWidget(self.tabWidget)

        # ---------------- Ethernet capture tab ----------------
        capture_tab = QtWidgets.QWidget()
        self.tabWidget.addTab(capture_tab, "Ethernet Capture")
        self.tab_capture_inst = tab_capture(
            capture_tab,
            backend=self.backend,
            initial_rt_file=str(getattr(self.rt, "txt_file", "") or ""),
            rt_apply_callback=getattr(MainWindow, "apply_rt_file", None),
        )

        # ---------------- Spectra tab (nested sub-tabs) ----------------
        spectra_tab = QtWidgets.QWidget()
        self.tabWidget.addTab(spectra_tab, "Spectra")

        spectra_layout = QtWidgets.QVBoxLayout(spectra_tab)
        self.spectraTabs = QtWidgets.QTabWidget()
        self.spectraTabs.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        spectra_layout.addWidget(self.spectraTabs)

        # ADC spectra (per TDC)
        adc_tab = QtWidgets.QWidget()
        self.spectraTabs.addTab(adc_tab, "ADC overall")
        self.tab_adc_spectra_inst = tab_adc_spectra(adc_tab, backend=self.backend, n_tdcs=40)

        # TDC spectra (fit-based real drift time per TDC)
        tdc_tab = QtWidgets.QWidget()
        self.spectraTabs.addTab(tdc_tab, "TDC overall")
        self.tab_tdc_spectra_inst = tab_tdc_spectra(tdc_tab, backend=self.backend, n_tdcs=40)

        # ADC channel spectra (single TDC, channels paged 2x3)
        adc_ch_tab = QtWidgets.QWidget()
        self.spectraTabs.addTab(adc_ch_tab, "ADC (per CH)")
        self.tab_adc_channels_inst = tab_adc_channels(
            adc_ch_tab, backend=self.backend, n_tdcs=40, n_channels=24
        )

        # TDC channel spectra (fit-based real drift time)
        tdc_ch_tab = QtWidgets.QWidget()
        self.spectraTabs.addTab(tdc_ch_tab, "TDC (per CH)")
        self.tab_tdc_channels_inst = tab_tdc_channels(
            tdc_ch_tab, backend=self.backend, n_tdcs=40, n_channels=24
        )

        hits_tab = QtWidgets.QWidget()
        self.spectraTabs.addTab(hits_tab, "CH Hits")
        self.tab_ch_hits_inst = tab_adc_channel_hits(hits_tab, backend=self.backend, n_tdcs=40)

        self.tab_fit_residual_overall_inst = []
        self.tab_fit_residuals_inst = []
        for g in self.geos:
            cid = int(getattr(g, "chamber_id", -1))
            is_active = getattr(g, "isActiveTDC", None)
            active_tdcs = []
            if is_active is not None:
                try:
                    active_tdcs = [t for t, flag in enumerate(is_active) if int(flag) == 1]
                except Exception:
                    active_tdcs = []

            residual_overall_tab = QtWidgets.QWidget()
            self.spectraTabs.addTab(residual_overall_tab, f"Residual overall C{cid}")
            self.tab_fit_residual_overall_inst.append(
                tab_fit_residual_overall(
                    residual_overall_tab,
                    backend=self.backend,
                    chamber_id=cid,
                )
            )

            residual_tab = QtWidgets.QWidget()
            self.spectraTabs.addTab(residual_tab, f"Residual C{cid}")
            self.tab_fit_residuals_inst.append(
                tab_fit_residual_spectra(
                    residual_tab,
                    backend=self.backend,
                    n_tdcs=40,
                    chamber_id=cid,
                    default_tdcs=active_tdcs,
                )
            )

        # ---------------- Geometry tabs: dynamic from geo list ----------------
        self.tab_geometries = []  # list of tab_geometry instances

        for g in self.geos:
            cid = int(g.chamber_id)
            tabw = QtWidgets.QWidget()
            self.tabWidget.addTab(tabw, f"Geometry {cid}")

            tab_inst = tab_geometry(
                tabw,
                g,
                default_filename="geometry.txt",
                backend=self.backend,
                chamber_id=cid,
                n_tdcs=40,
                rt=self.rt
            )
            self.tab_geometries.append(tab_inst)

            # When capture restarts, clear event caches (so colors don't mismatch)
            if self.backend is not None and hasattr(self.backend, "capture_started"):
                self.backend.capture_started.connect(self.backend.clear_event_cache)

        # ---------------- clear info button ----------------
        self.pushButton_clearinfo = QtWidgets.QPushButton()
        self.pushButton_clearinfo.setText("clear info")
        self.pushButton_clearinfo.clicked.connect(self.clear_info)
        mainLayout.addWidget(self.pushButton_clearinfo)

        label = QtWidgets.QLabel()
        label.setText("Output Log")
        label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        mainLayout.addWidget(label)

        self.textBrowser = QtWidgets.QTextBrowser()
        self.textBrowser.setMinimumHeight(140)
        self.textBrowser.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.Preferred,
        )
        mainLayout.addWidget(self.textBrowser)

    def clear_info(self):
        self.textBrowser.clear()
