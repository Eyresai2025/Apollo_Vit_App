import sys
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import Counter, defaultdict

from PyQt5 import QtCore, QtGui, QtWidgets, QtPrintSupport # type: ignore


# ----------------------------
# Data Model (replace with DB later)
# ----------------------------
@dataclass
class InspectionRecord:
    ts: datetime
    sku: str
    result: str          # "Good" / "Bad"
    category: str        # "OE" / "Replacement" / "Scrap"
    defect: str          # e.g. "Tread blisters"
    tyre_size: str       # e.g. "R13", "R14"


class InspectionDataStore:
    """
    Demo store. Replace `load_records()` with Mongo/SQL fetch.
    """
    def __init__(self):
        self.records = []

    def load_demo_records(self, days=10, n=2500):
        skus = ["SKU-A", "SKU-B", "SKU-C", "SKU-D", "SKU-E"]
        defects = [
            "Tread blisters", "Segment flash", "Shoulder crack", "Sidewall flash",
            "Bead ring flash", "Bead toe flash", "Bead crack", "Barcode imperfection",
            "Bead blister", "Inner liner crack", "No defect"
        ]
        tyre_sizes = ["R13", "R14", "R15", "R16", "R17"]
        categories = ["OE", "Replacement", "Scrap"]

        now = datetime.now()
        start = now - timedelta(days=days)

        self.records.clear()
        for _ in range(n):
            ts = start + timedelta(seconds=random.randint(0, int((now - start).total_seconds())))
            sku = random.choice(skus)
            tyre_size = random.choice(tyre_sizes)

            # Make "Bad" less common
            is_bad = random.random() < 0.12
            result = "Bad" if is_bad else "Good"

            category = random.choices(categories, weights=[0.6, 0.3, 0.1])[0]

            # defect only if bad mostly
            if is_bad:
                defect = random.choice(defects[:-1])
            else:
                defect = "No defect"

            self.records.append(InspectionRecord(ts, sku, result, category, defect, tyre_size))

    def available_skus(self):
        return sorted({r.sku for r in self.records})

    def query(self, dt_from: datetime, dt_to: datetime, sku: str | None):
        out = []
        for r in self.records:
            if dt_from <= r.ts <= dt_to:
                if (sku is None) or (r.sku == sku):
                    out.append(r)
        return out

    @staticmethod
    def aggregate(records):
        total = len(records)
        good = sum(1 for r in records if r.result == "Good")
        bad = total - good
        sku_count = len({r.sku for r in records})

        cat_counts = Counter(r.category for r in records)
        tyre_counts = Counter(r.tyre_size for r in records)

        # defects only for bad
        defect_counts = Counter(r.defect for r in records if r.result == "Bad" and r.defect != "No defect")
        top_defects = defect_counts.most_common(10)

        # SKU-wise counts
        sku_counts = Counter(r.sku for r in records)

        return {
            "total": total,
            "good": good,
            "bad": bad,
            "sku_count": sku_count,
            "cat_counts": cat_counts,
            "tyre_counts": tyre_counts,
            "top_defects": top_defects,
            "sku_counts": sku_counts,
        }


# ----------------------------
# UI: Card Widgets (white 3D style)
# ----------------------------
class CardWidget(QtWidgets.QFrame):
    def __init__(self, parent=None, radius=18):
        super().__init__(parent)
        self.setObjectName("CardWidget")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setStyleSheet(f"""
            #CardWidget {{
                background: #ffffff;
                border-radius: {radius}px;
                border: 1px solid #eef2f7;
            }}
        """)

        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 10)
        shadow.setColor(QtGui.QColor(15, 23, 42, 28))  # subtle shadow
        self.setGraphicsEffect(shadow)


class StatCard(CardWidget):
    def __init__(self, title: str, value: str, subtitle: str = "", accent="#571c86", parent=None):
        super().__init__(parent=parent, radius=18)
        self._accent = accent

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        self.lbl_title = QtWidgets.QLabel(title)
        self.lbl_title.setStyleSheet("color:#475569; font-size:12px; font-weight:600;")
        layout.addWidget(self.lbl_title)

        self.lbl_value = QtWidgets.QLabel(value)
        self.lbl_value.setStyleSheet(f"color:#0f172a; font-size:26px; font-weight:800;")
        layout.addWidget(self.lbl_value)

        self.lbl_sub = QtWidgets.QLabel(subtitle)
        self.lbl_sub.setStyleSheet("color:#64748b; font-size:11px;")
        layout.addWidget(self.lbl_sub)

        # Accent bar
        bar = QtWidgets.QFrame()
        bar.setFixedHeight(4)
        bar.setStyleSheet(f"background:{self._accent}; border-radius:2px;")
        layout.addWidget(bar)

        layout.addStretch(1)

    def set_value(self, value: str, subtitle: str = ""):
        self.lbl_value.setText(value)
        self.lbl_sub.setText(subtitle)


class ListCard(CardWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent=parent, radius=18)
        self.title = QtWidgets.QLabel(title)
        self.title.setStyleSheet("color:#0f172a; font-size:13px; font-weight:800;")

        self.container = QtWidgets.QVBoxLayout(self)
        self.container.setContentsMargins(16, 16, 16, 16)
        self.container.setSpacing(10)

        self.container.addWidget(self.title)

        self.list_area = QtWidgets.QVBoxLayout()
        self.list_area.setSpacing(8)
        self.container.addLayout(self.list_area)
        self.container.addStretch(1)

    def set_items(self, items: list[tuple[str, int]], empty_text="No data"):
        # clear
        while self.list_area.count():
            item = self.list_area.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if not items:
            lbl = QtWidgets.QLabel(empty_text)
            lbl.setStyleSheet("color:#64748b; font-size:11px;")
            self.list_area.addWidget(lbl)
            return

        for name, count in items:
            row = QtWidgets.QHBoxLayout()
            left = QtWidgets.QLabel(name)
            left.setStyleSheet("color:#334155; font-size:11px; font-weight:600;")
            right = QtWidgets.QLabel(str(count))
            right.setStyleSheet("color:#0f172a; font-size:11px; font-weight:800;")
            row.addWidget(left)
            row.addStretch(1)
            row.addWidget(right)

            wrap = QtWidgets.QWidget()
            wrap.setLayout(row)
            self.list_area.addWidget(wrap)


class TimeCard(CardWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent, radius=18)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(6)

        self.lbl_time = QtWidgets.QLabel("--:--")
        self.lbl_time.setStyleSheet("color:#0f172a; font-size:26px; font-weight:900; letter-spacing:1px;")
        lay.addWidget(self.lbl_time)

        self.lbl_date = QtWidgets.QLabel("—")
        self.lbl_date.setStyleSheet("color:#475569; font-size:12px; font-weight:700;")
        lay.addWidget(self.lbl_date)

        lay.addStretch(1)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)
        self._tick()

    def _tick(self):
        now = datetime.now()
        self.lbl_time.setText(now.strftime("%I:%M %p").lstrip("0"))
        self.lbl_date.setText(now.strftime("%A, %d %b %Y"))

# ----------------------------
# Filter Bar (NO CSS for DateTime + SKU combo)
# - DateTimeEdit + ComboBox will use default OS/Qt style
# - Only Apply/PDF buttons are styled (optional)
# ----------------------------
class FilterBar(CardWidget):
    applied = QtCore.pyqtSignal(datetime, datetime, object)  # (from, to, sku or None)
    export_pdf = QtCore.pyqtSignal()

    def __init__(self, skus: list[str], parent=None):
        super().__init__(parent=parent, radius=18)

        # Make bar shorter
        self.setFixedHeight(56)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(10)

        title = QtWidgets.QLabel("Filters")
        title.setStyleSheet("color:#0f172a; font-size:12px; font-weight:900;")
        lay.addWidget(title)

        # Helper labels (small, clean)
        def mk_label(text: str, strong=False):
            l = QtWidgets.QLabel(text)
            if strong:
                l.setStyleSheet("color:#0f172a; font-size:11px; font-weight:900;")
            else:
                l.setStyleSheet("color:#475569; font-size:11px; font-weight:800;")
            return l

        lay.addSpacing(6)

        # ----------- DEFAULT widgets (NO CSS) -----------
        self.dt_from = QtWidgets.QDateTimeEdit()
        self.dt_from.setCalendarPopup(True)
        self.dt_from.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.dt_from.setFixedSize(150, 30)

        self.dt_to = QtWidgets.QDateTimeEdit()
        self.dt_to.setCalendarPopup(True)
        self.dt_to.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.dt_to.setFixedSize(150, 30)

        self.cmb_sku = QtWidgets.QComboBox()
        self.cmb_sku.setFixedSize(140, 30)
        self.cmb_sku.addItem("All SKUs")
        self.cmb_sku.addItems(skus)

        # ----------- Buttons (you can keep style) -----------
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_apply.setFixedSize(64, 30)
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background:#571c86; color:#fff;
                border:none; border-radius:10px;
                font-weight:900; font-size:11px;
                padding:6px 12px;
            }
            QPushButton:hover { background:#4a1573; }
            QPushButton:pressed { background:#3f125f; }
        """)

        self.btn_export = QtWidgets.QPushButton("Generate Report (PDF)")
        self.btn_export.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_export.setFixedHeight(30)
        self.btn_export.setStyleSheet("""
            QPushButton {
                background:#0f172a; color:#fff;
                border:none; border-radius:10px;
                font-weight:900; font-size:11px;
                padding:6px 12px;
            }
            QPushButton:hover { background:#111c33; }
            QPushButton:pressed { background:#0b1324; }
        """)

        # Layout
        lay.addWidget(mk_label("From:"))
        lay.addWidget(self.dt_from)

        lay.addWidget(mk_label("To:"))
        lay.addWidget(self.dt_to)

        lay.addSpacing(10)
        lay.addWidget(mk_label("SKU:", strong=True))
        lay.addWidget(self.cmb_sku)

        lay.addStretch(1)
        lay.addWidget(self.btn_apply)
        lay.addWidget(self.btn_export)

        # Signals
        self.btn_apply.clicked.connect(self._emit_apply)
        self.btn_export.clicked.connect(self.export_pdf.emit)

    def set_range(self, dt_from: datetime, dt_to: datetime):
        self.dt_from.setDateTime(QtCore.QDateTime(dt_from))
        self.dt_to.setDateTime(QtCore.QDateTime(dt_to))

    def _emit_apply(self):
        dt_from = self.dt_from.dateTime().toPyDateTime()
        dt_to = self.dt_to.dateTime().toPyDateTime()
        if dt_from > dt_to:
            QtWidgets.QMessageBox.warning(self, "Invalid Range", "From datetime cannot be after To datetime.")
            return
        sku_txt = self.cmb_sku.currentText()
        sku = None if sku_txt == "All SKUs" else sku_txt
        self.applied.emit(dt_from, dt_to, sku)


# ----------------------------
# Dashboard Page (cards grid)
# ----------------------------
class DashboardCardsPage(QtWidgets.QWidget):
    def __init__(self, store: InspectionDataStore, title: str, default_range: tuple[datetime, datetime], parent=None):
        super().__init__(parent)
        self.store = store
        self.page_title = title

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)


        # Filter bar
        self.filter_bar = FilterBar(self.store.available_skus())
        root.addWidget(self.filter_bar)

        # Scroll area for cards
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        root.addWidget(scroll, 1)

        body = QtWidgets.QWidget()
        scroll.setWidget(body)

        self.grid = QtWidgets.QGridLayout(body)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(14)

        # --- Cards ---
        self.card_total = StatCard("Total Tyres Inspected", "0", "—", accent="#3b82f6")
        self.card_good = StatCard("Good Tyres", "0", "—", accent="#10b981")
        self.card_bad = StatCard("Bad Tyres", "0", "—", accent="#ef4444")
        self.card_skus = StatCard("SKUs", "0", "Unique SKUs", accent="#8b5cf6")

        self.card_oe = StatCard("OE Count", "0", "—", accent="#f97316")
        self.card_rep = StatCard("Replacement Count", "0", "—", accent="#06b6d4")
        self.card_scrap = StatCard("Scrap Count", "0", "—", accent="#334155")

        self.card_top_defects = ListCard("Top Defects")
        self.card_tyre_sizes = ListCard("Tyre Size Counts")
        self.card_sku_wise = ListCard("SKU-wise Counts")

        # Place in grid (nice layout)
        self.grid.addWidget(self.card_total, 0, 0)
        self.grid.addWidget(self.card_good, 0, 1)
        self.grid.addWidget(self.card_bad, 0, 2)
        self.grid.addWidget(self.card_skus, 0, 3)

        self.grid.addWidget(self.card_oe, 1, 0)
        self.grid.addWidget(self.card_rep, 1, 1)
        self.grid.addWidget(self.card_scrap, 1, 2)
        # keep 1,3 empty to reduce clutter; or add another card later

        self.grid.addWidget(self.card_top_defects, 2, 0, 1, 2)   # span 2 columns
        self.grid.addWidget(self.card_tyre_sizes, 2, 2, 1, 2)    # span 2 columns

        self.grid.addWidget(self.card_sku_wise, 3, 0, 1, 4)      # full width

        # Default range
        self.dt_from, self.dt_to = default_range
        self.sku = None
        self.filter_bar.set_range(self.dt_from, self.dt_to)

        self.filter_bar.applied.connect(self.apply_filters)
        self.filter_bar.export_pdf.connect(self.export_report_pdf)

        # Initial load
        self.refresh_cards()

    def apply_filters(self, dt_from: datetime, dt_to: datetime, sku):
        self.dt_from, self.dt_to, self.sku = dt_from, dt_to, sku
        self.refresh_cards()

    def refresh_cards(self):
        records = self.store.query(self.dt_from, self.dt_to, self.sku)
        agg = self.store.aggregate(records)

        self.card_total.set_value(str(agg["total"]), f"{self.dt_from:%Y-%m-%d %H:%M} → {self.dt_to:%Y-%m-%d %H:%M}")
        self.card_good.set_value(str(agg["good"]), "Pass / Good")
        self.card_bad.set_value(str(agg["bad"]), "Fail / Bad")
        self.card_skus.set_value(str(agg["sku_count"]), "Unique SKUs")

        self.card_oe.set_value(str(agg["cat_counts"].get("OE", 0)), "OE")
        self.card_rep.set_value(str(agg["cat_counts"].get("Replacement", 0)), "Replacement")
        self.card_scrap.set_value(str(agg["cat_counts"].get("Scrap", 0)), "Scrap")

        self.card_top_defects.set_items(agg["top_defects"], empty_text="No bad defects in this range.")
        self.card_tyre_sizes.set_items(sorted(agg["tyre_counts"].items()), empty_text="No tyre sizes found.")
        self.card_sku_wise.set_items(sorted(agg["sku_counts"].items(), key=lambda x: x[1], reverse=True), empty_text="No SKU data.")

    def export_report_pdf(self):
        # Ask where to save
        default_name = f"Apollo_Dashboard_Report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Report as PDF", default_name, "PDF Files (*.pdf)")
        if not path:
            return

        # Build simple HTML report
        sku_txt = self.sku if self.sku else "All SKUs"
        records = self.store.query(self.dt_from, self.dt_to, self.sku)
        agg = self.store.aggregate(records)

        def ul(items):
            if not items:
                return "<p style='color:#64748b;'>No data</p>"
            li = "".join([f"<li><b>{name}</b>: {count}</li>" for name, count in items])
            return f"<ul>{li}</ul>"

        html = f"""
        <html>
          <head>
            <meta charset="utf-8" />
          </head>
          <body style="font-family:Segoe UI, Arial; color:#0f172a;">
            <h2 style="margin:0;">Apollo Dashboard Report</h2>
            <p style="margin:6px 0; color:#475569;">
              Page: <b>{self.page_title}</b><br/>
              Range: <b>{self.dt_from:%Y-%m-%d %H:%M}</b> to <b>{self.dt_to:%Y-%m-%d %H:%M}</b><br/>
              SKU: <b>{sku_txt}</b>
            </p>
            <hr/>

            <h3>Summary</h3>
            <ul>
              <li><b>Total Tyres Inspected</b>: {agg["total"]}</li>
              <li><b>Good Tyres</b>: {agg["good"]}</li>
              <li><b>Bad Tyres</b>: {agg["bad"]}</li>
              <li><b>Unique SKUs</b>: {agg["sku_count"]}</li>
            </ul>

            <h3>Category Counts</h3>
            <ul>
              <li><b>OE</b>: {agg["cat_counts"].get("OE", 0)}</li>
              <li><b>Replacement</b>: {agg["cat_counts"].get("Replacement", 0)}</li>
              <li><b>Scrap</b>: {agg["cat_counts"].get("Scrap", 0)}</li>
            </ul>

            <h3>Top Defects (Bad Only)</h3>
            {ul(agg["top_defects"])}

            <h3>Tyre Size Counts</h3>
            {ul(sorted(agg["tyre_counts"].items()))}

            <h3>SKU-wise Counts</h3>
            {ul(sorted(agg["sku_counts"].items(), key=lambda x: x[1], reverse=True))}
          </body>
        </html>
        """

        doc = QtGui.QTextDocument()
        doc.setHtml(html)

        printer = QtPrintSupport.QPrinter(QtPrintSupport.QPrinter.HighResolution)
        printer.setOutputFormat(QtPrintSupport.QPrinter.PdfFormat)
        printer.setOutputFileName(path)
        printer.setPageSize(QtPrintSupport.QPrinter.A4)

        doc.print_(printer)

        QtWidgets.QMessageBox.information(self, "Report Saved", f"PDF saved to:\n{path}")


# ----------------------------
# Main Window (Today / Previous switch)
# ----------------------------
class DashboardWindow(QtWidgets.QMainWindow):
    def __init__(self, store: InspectionDataStore):
        super().__init__()
        self.store = store
        self.setWindowTitle("Apollo Tyres - Dashboard")
        self.resize(1300, 820)

        # App background
        central = QtWidgets.QWidget()
        central.setStyleSheet("background:#f8fafc;")
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Header: segmented buttons
        header = QtWidgets.QHBoxLayout()
        header.setSpacing(10)

        brand = QtWidgets.QLabel("Apollo Dashboard")
        brand.setStyleSheet("color:#0f172a; font-size:18px; font-weight:900;")
        header.addWidget(brand)

        header.addStretch(1)

        self.btn_today = QtWidgets.QPushButton("Today Inspection")
        self.btn_prev = QtWidgets.QPushButton("Previous Inspection")

        for b in (self.btn_today, self.btn_prev):
            b.setCursor(QtCore.Qt.PointingHandCursor)
            b.setCheckable(True)
            b.setStyleSheet("""
                QPushButton {
                    background:#ffffff; color:#0f172a;
                    border:1px solid #e2e8f0;
                    padding:8px 14px; border-radius:12px;
                    font-weight:900;
                }
                QPushButton:checked {
                    background:#571c86; color:#ffffff;
                    border:1px solid #571c86;
                }
            """)

        header.addWidget(self.btn_today)
        header.addWidget(self.btn_prev)

        root.addLayout(header)

        # Pages
        self.stack = QtWidgets.QStackedWidget()
        root.addWidget(self.stack, 1)

        now = datetime.now()
        start_today = datetime(now.year, now.month, now.day, 0, 0, 0)

        today_page = DashboardCardsPage(
            store=self.store,
            title="Today Inspection",
            default_range=(start_today, now),
        )

        prev_page = DashboardCardsPage(
            store=self.store,
            title="Previous Inspection",
            default_range=(now - timedelta(days=7), now),
        )

        self.stack.addWidget(today_page)
        self.stack.addWidget(prev_page)

        # Switch wiring
        self.btn_today.clicked.connect(lambda: self.set_page(0))
        self.btn_prev.clicked.connect(lambda: self.set_page(1))

        self.set_page(0)

    def set_page(self, idx: int):
        self.stack.setCurrentIndex(idx)
        self.btn_today.setChecked(idx == 0)
        self.btn_prev.setChecked(idx == 1)



class ApolloDashboardCardsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, store=None):
        super().__init__(parent)

        # If you don't pass store, it will use demo data
        if store is None:
            store = InspectionDataStore()
            store.load_demo_records(days=15, n=3500)

        self.store = store

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        now = datetime.now()
        start_today = datetime(now.year, now.month, now.day, 0, 0, 0)

        # Embedded switcher like tabs
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(18, 10, 18, 10)

        self.btn_today = QtWidgets.QPushButton("Today Inspection")
        self.btn_prev  = QtWidgets.QPushButton("Previous Inspection")
        for b in (self.btn_today, self.btn_prev):
            b.setCheckable(True)
            b.setCursor(QtCore.Qt.PointingHandCursor)
            b.setStyleSheet("""
                QPushButton{background:#fff;color:#0f172a;border:1px solid #e2e8f0;
                            padding:8px 14px;border-radius:12px;font-weight:900;}
                QPushButton:checked{background:#571c86;color:#fff;border:1px solid #571c86;}
            """)

        header.addWidget(self.btn_today)
        header.addWidget(self.btn_prev)
        header.addStretch(1)

        root.addLayout(header)

        self.stack = QtWidgets.QStackedWidget()
        root.addWidget(self.stack, 1)

        self.today_page = DashboardCardsPage(self.store, "Today Inspection", (start_today, now))
        self.prev_page  = DashboardCardsPage(self.store, "Previous Inspection", (now - timedelta(days=7), now))

        self.stack.addWidget(self.today_page)
        self.stack.addWidget(self.prev_page)

        self.btn_today.clicked.connect(lambda: self._set_page(0))
        self.btn_prev.clicked.connect(lambda: self._set_page(1))
        self._set_page(0)

    def _set_page(self, idx):
        self.stack.setCurrentIndex(idx)
        self.btn_today.setChecked(idx == 0)
        self.btn_prev.setChecked(idx == 1)
