import os
import sys

from PyQt6.QtWidgets import QApplication

from ui.main_window import WhiteboardStudioWindow


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets")

    app = QApplication(sys.argv)
    app.setApplicationName("Whiteboard Studio Pro")
    win = WhiteboardStudioWindow(assets_dir=assets_dir)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
