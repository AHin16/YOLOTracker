import json
from pathlib import Path
import re

from PySide6.QtCore import QCoreApplication, QLibraryInfo, QObject, QTranslator, Qt, Signal
from PySide6.QtWidgets import QAbstractButton, QComboBox, QGroupBox, QLabel, QTabWidget, QTableWidget, QWidget


with (Path(__file__).parent / "translations" / "zh_CN.json").open(encoding="utf-8") as catalog_file:
    TRANSLATIONS = json.load(catalog_file)


def _template_pattern(template):
    parts = re.split(r"(\{[a-z_]+\})", template)
    pattern_parts = []
    for part in parts:
        if re.fullmatch(r"\{[a-z_]+\}", part):
            name = part[1:-1]
            value_pattern = r"[^；\n]+?" if name == "metric" else r".+?"
            if name in ("lower", "upper", "bound") or (name == "value" and "{bound}" in template):
                value_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
            pattern_parts.append(f"(?P<{name}>{value_pattern})")
        else:
            pattern_parts.append(re.escape(part))
    return re.compile("".join(pattern_parts), re.DOTALL)


class LanguageManager(QObject):
    changed = Signal(str)

    def __init__(self, language="zh_CN", parent=None):
        super().__init__(parent)
        self.language = "en"
        self._properties = {}
        self._tabs = []
        self._combos = []
        self._tables = []
        self._registered = set()
        self._aliases = {chinese: english for english, chinese in TRANSLATIONS.items()}
        self._templates = [
            (_template_pattern(source), english, chinese)
            for english, chinese in TRANSLATIONS.items() if re.search(r"\{[a-z_]+\}", english)
            for source in (english, chinese)
        ]
        self._qt_translator = QTranslator(self)
        self._qt_translator.load("qtbase_zh_CN", QLibraryInfo.path(QLibraryInfo.TranslationsPath))
        self.set_language(language)

    def text(self, source):
        source = str(source)
        canonical = self._aliases.get(source, source)
        if canonical in TRANSLATIONS:
            return TRANSLATIONS[canonical] if self.language == "zh_CN" else canonical
        for pattern, english, chinese in self._templates:
            match = pattern.fullmatch(source)
            if match:
                values = match.groupdict()
                for key in ("state", "phase", "status", "description", "reason", "metric"):
                    if key in values and values[key] != source:
                        values[key] = self.text(values[key])
                return (chinese if self.language == "zh_CN" else english).format(**values)
        for separator, localized_separator in (("\n", "\n"), ("；", "; ")):
            if separator in source:
                joiner = localized_separator if self.language == "en" else separator
                return joiner.join(self.text(part) for part in source.split(separator))
        return source

    def set_text(self, widget, source, **values):
        self.set_property(widget, "text", source, **values)

    def set_property(self, widget, name, source, **values):
        self._properties[(widget, name)] = (source, values)
        widget.setProperty(name, self._format(source, values))

    def _format(self, source, values):
        translated = self.text(source)
        values = {key: self.text(value) if key in ("field", "kind") else value for key, value in values.items()}
        return translated.format(**values) if values else translated

    def forget_text(self, widget):
        self._properties.pop((widget, "text"), None)

    def register(self, root):
        for widget in [root, *root.findChildren(QWidget)]:
            if widget in self._registered:
                continue
            self._registered.add(widget)
            properties = ["toolTip", "accessibleName", "placeholderText"]
            if isinstance(widget, (QLabel, QAbstractButton)):
                properties.append("text")
            if isinstance(widget, QGroupBox):
                properties.append("title")
            for name in properties:
                source = widget.property(name)
                if source in TRANSLATIONS or source in self._aliases:
                    self.set_property(widget, name, source)
            if isinstance(widget, QTabWidget):
                for index in range(widget.count()):
                    source = widget.tabText(index)
                    self._tabs.append((widget, index, source))
                    widget.setTabText(index, self.text(source))
            if isinstance(widget, QComboBox) and widget.objectName() != "languageCombo":
                sources = [widget.itemText(index) for index in range(widget.count())]
                self._combos.append((widget, sources))
                for index, source in enumerate(sources):
                    if widget.itemData(index) is None:
                        widget.setItemData(index, source)
            if isinstance(widget, QTableWidget):
                self._tables.append(widget)
                for column in range(widget.columnCount()):
                    item = widget.horizontalHeaderItem(column)
                    if item is not None:
                        self.set_item_text(item, item.text())
        self._refresh_items()

    def set_item_text(self, item, source):
        item.setData(Qt.UserRole + 1, source)
        item.setText(self.text(source))
        item.setToolTip(self.text(source))

    def _refresh_items(self):
        for combo, sources in self._combos:
            blocked = combo.blockSignals(True)
            for index, source in enumerate(sources):
                display = {"botsort": "BoT-SORT", "bytetrack": "ByteTrack", "auto": "Auto",
                           "cpu": "CPU", "cuda:0": "GPU (CUDA:0)"}.get(source, source)
                combo.setItemText(index, self.text(source) if self.language == "zh_CN" else display)
            combo.blockSignals(blocked)
        for table in self._tables:
            blocked = table.blockSignals(True)
            items = [table.horizontalHeaderItem(column) for column in range(table.columnCount())]
            items.extend(table.item(row, column) for row in range(table.rowCount()) for column in range(table.columnCount()))
            for item in items:
                if item is not None and item.data(Qt.UserRole + 1) is not None:
                    item.setText(self.text(item.data(Qt.UserRole + 1)))
                    item.setToolTip(item.text())
            table.blockSignals(blocked)

    def set_language(self, language):
        language = language if language in ("zh_CN", "en") else "zh_CN"
        if language == self.language:
            return
        self.language = language
        app = QCoreApplication.instance()
        if app is not None:
            if language == "zh_CN":
                app.installTranslator(self._qt_translator)
            else:
                app.removeTranslator(self._qt_translator)
        for (widget, name), (source, values) in self._properties.items():
            widget.setProperty(name, self._format(source, values))
        for widget, index, source in self._tabs:
            widget.setTabText(index, self.text(source))
        self._refresh_items()
        self.changed.emit(language)
