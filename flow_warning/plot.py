from collections import deque

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget
from ui_language import LanguageManager


class MetricPlot(QWidget):
    def __init__(self, parent=None, language=None):
        super().__init__(parent)
        self.language = language or LanguageManager(parent=self)
        self.language.changed.connect(self.update)
        self.values = deque(maxlen=240)
        self.limits = None
        self.setMinimumHeight(72)
        self.setAccessibleName("指标趋势，浅色区域为正常基线范围")

    def append(self, timestamp, value):
        self.values.append((timestamp, value))
        self.update()

    def reset(self):
        self.values.clear()
        self.limits = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), self.palette().base())
        chart = QRectF(42, 6, max(self.width() - 48, 1), max(self.height() - 24, 1))
        values = [value for _, value in self.values if value is not None]
        if self.limits:
            values.extend([self.limits["lower"], self.limits["upper"]])
        if not values:
            painter.setPen(self.palette().text().color())
            painter.drawText(self.rect(), Qt.AlignCenter, self.language.text("等待完整时间窗口"))
            return
        lower, upper = min(values), max(values)
        padding = max((upper - lower) * 0.1, abs(upper) * 0.02, 0.001)
        lower -= padding
        upper += padding

        def vertical(value):
            return chart.bottom() - (value - lower) / (upper - lower) * chart.height()

        if self.limits:
            top = vertical(self.limits["upper"])
            bottom = vertical(self.limits["lower"])
            painter.fillRect(QRectF(chart.left(), top, chart.width(), bottom - top), QColor(62, 160, 105, 45))
        painter.setPen(self.palette().text().color())
        painter.drawText(QRectF(0, 0, 38, 16), Qt.AlignRight, f"{upper:.2g}")
        painter.drawText(QRectF(0, chart.bottom() - 12, 38, 16), Qt.AlignRight, f"{lower:.2g}")
        if self.values:
            painter.drawText(QRectF(chart.left(), chart.bottom() + 2, chart.width(), 16), Qt.AlignRight,
                             f"{self.values[-1][0]:.2f} s")
        painter.setPen(QPen(QColor(35, 140, 195), 2))
        segment = []
        for index, (_, value) in enumerate(self.values):
            if value is None:
                if len(segment) > 1:
                    painter.drawPolyline(QPolygonF(segment))
                segment = []
                continue
            horizontal = chart.left() + index / max(len(self.values) - 1, 1) * chart.width()
            point = QPointF(horizontal, vertical(value))
            segment.append(point)
            painter.drawEllipse(point, 1.5, 1.5)
        if len(segment) > 1:
            painter.drawPolyline(QPolygonF(segment))
