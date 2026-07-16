from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QButtonGroup
import sys

class TestDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.resize(600, 200)
        self.setStyleSheet("background-color: #1e1e1e;")
        layout = QVBoxLayout(self)
        
        state_layout = QHBoxLayout()
        self.btn_group = QButtonGroup(self)
        
        self.btn_baseline = QPushButton("BASELINE\n(Rollback)")
        self.btn_opt = QPushButton("OPTIMIZED\n(Apply)")
        
        for btn in [self.btn_baseline, self.btn_opt]:
            btn.setCheckable(True)
            btn.setMinimumHeight(100)
            self.btn_group.addButton(btn)
            
        self.btn_baseline.setChecked(True)
        
        style = """
        QPushButton {
            font-size: 24px;
            font-weight: bold;
            color: #aaaaaa;
            background-color: #333333;
            border: 4px solid #444444;
            border-radius: 12px;
        }
        QPushButton:checked {
            color: #ffffff;
            background-color: #d84315;
            border: 6px solid #ff9800;
        }
        """
        self.setStyleSheet(style)
        
        state_layout.addWidget(self.btn_baseline)
        state_layout.addWidget(self.btn_opt)
        layout.addLayout(state_layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    d = TestDialog()
    d.show()
    sys.exit(app.exec())
