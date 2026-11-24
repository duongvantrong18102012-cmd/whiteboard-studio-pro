STUDIO_QSS = """
QMainWindow, QWidget {
    background-color: #0F1117;
    color: #E6EAF2;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}

/* Top Toolbar */
QFrame#TopBar {
    background-color: #161923;
    border-bottom: 1px solid #242938;
}

QLabel#AppBrand {
    font-size: 15px;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: 0.3px;
}

QLabel#StatusPill {
    background-color: #1E2433;
    color: #94A3B8;
    border: 1px solid #2B3348;
    border-radius: 12px;
    padding: 4px 12px;
    font-size: 12px;
}

/* Preview & Stage */
QFrame#StageBox {
    background-color: #090B10;
    border: 1px solid #222736;
    border-radius: 10px;
}

QFrame#TransportBar {
    background-color: #151822;
    border: 1px solid #232838;
    border-radius: 8px;
}

/* Inspector Panel & Tabs */
QFrame#InspectorPanel {
    background-color: #141720;
    border-left: 1px solid #232838;
}

QTabWidget::pane {
    border: none;
    background-color: #141720;
}

QTabBar::tab {
    background-color: #1B1F2C;
    color: #94A3B8;
    border: 1px solid #262C3E;
    border-radius: 6px;
    padding: 7px 10px;
    margin: 4px 3px;
    font-weight: 600;
    font-size: 12px;
}

QTabBar::tab:selected {
    background-color: #2563EB;
    color: #FFFFFF;
    border-color: #3B82F6;
}

QTabBar::tab:hover:!selected {
    background-color: #232838;
    color: #E2E8F0;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

/* Cards inside Inspector */
QFrame#Card {
    background-color: #1A1E2A;
    border: 1px solid #272D3F;
    border-radius: 8px;
}

QLabel#SectionTitle {
    color: #93C5FD;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

QLabel#MutedLabel {
    color: #94A3B8;
    font-size: 12px;
}

QLabel#ValueBadge {
    background-color: #252B3D;
    color: #E2E8F0;
    border-radius: 4px;
    padding: 2px 7px;
    font-family: 'Consolas', monospace;
    font-size: 12px;
    font-weight: 600;
}

/* Buttons */
QPushButton {
    background-color: #242A3B;
    color: #E2E8F0;
    border: 1px solid #323A52;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
    min-height: 22px;
}

QPushButton:hover {
    background-color: #2E364D;
    border-color: #475375;
}

QPushButton:pressed {
    background-color: #1E2332;
}

QPushButton:disabled {
    background-color: #181B24;
    color: #525B70;
    border-color: #222634;
}

QPushButton#primary {
    background-color: #2563EB;
    border-color: #3B82F6;
    color: #FFFFFF;
}
QPushButton#primary:hover { background-color: #3B82F6; }

QPushButton#accent {
    background-color: #D97706;
    border-color: #F59E0B;
    color: #FFFFFF;
}
QPushButton#accent:hover { background-color: #F59E0B; }

QPushButton#success {
    background-color: #059669;
    border-color: #10B981;
    color: #FFFFFF;
    padding: 7px 16px;
}
QPushButton#success:hover { background-color: #10B981; }

QPushButton#danger {
    background-color: #991B1B;
    border-color: #DC2626;
    color: #FEE2E2;
}
QPushButton#danger:hover { background-color: #DC2626; color: #FFFFFF; }

QPushButton#modeBtn {
    text-align: left;
    padding: 7px 10px;
    background-color: #161A25;
    border: 1px solid #2A3144;
    border-radius: 6px;
    font-weight: 600;
    color: #CBD5E1;
}
QPushButton#modeBtn:checked {
    background-color: #1E3A8A;
    border-color: #3B82F6;
    color: #FFFFFF;
}

/* Hand Card */
QFrame#HandCard {
    background-color: #151923;
    border: 2px solid #272D3F;
    border-radius: 8px;
}
QFrame#HandCard:hover {
    border-color: #4B5563;
}
QFrame#HandCard[selected="true"] {
    background-color: #1B2845;
    border: 2px solid #3B82F6;
}

/* Inputs & Lists */
QListWidget {
    background-color: #141822;
    border: 1px solid #272D3F;
    border-radius: 6px;
    padding: 3px;
    outline: none;
}
QListWidget::item {
    padding: 6px 8px;
    border-radius: 5px;
    margin-bottom: 2px;
}
QListWidget::item:selected {
    background-color: #1E3A8A;
    color: #FFFFFF;
}
QListWidget::item:hover:!selected {
    background-color: #1F2536;
}

QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #141822;
    border: 1px solid #2E354A;
    border-radius: 6px;
    padding: 4px 8px;
    color: #F1F5F9;
    min-height: 24px;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #475569;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #272D3F;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #3B82F6;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #F8FAFC;
    border: 2px solid #3B82F6;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 8px;
}

QCheckBox {
    spacing: 8px;
    color: #CBD5E1;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #475569;
    background: #141822;
}
QCheckBox::indicator:checked {
    background: #2563EB;
    border-color: #3B82F6;
}

QProgressBar {
    border: 1px solid #2A3144;
    border-radius: 6px;
    background-color: #141822;
    text-align: center;
    color: #F8FAFC;
    font-weight: 600;
    min-height: 18px;
}
QProgressBar::chunk {
    background-color: #10B981;
    border-radius: 5px;
}

QScrollBar:vertical {
    background: #12151E;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #2E354A;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""
