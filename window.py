# window.py
import csv
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from pathlib import Path


class WindowDraw:
    """
    Simple OpenCV drawing window:
    - Left-click to draw white strokes on black background.
    - Right-click to erase (same brush).
    - 'c' to clear canvas.
    - 's' to save the drawing (28x28 PNG + CSV).
    - 'q' or close window to exit.

    Saved PNGs go to: customFigures/original/{name}_28x28.png
    """

    def __init__(self, name: str = "Draw", width_height: int = 128, brush_size: int = 4) -> None:
        self.width, self.height = width_height, width_height
        self.window_name = name
        self.drawing = False
        self.last_pt = None
        self.brush_size = brush_size
        self.min_brush_size = 0
        self.max_brush_size = 8
        self.brush_colour = 0xFF               # white
        self.screen_colour = self.brush_colour ^ 0xFF  # black
        self.erase_colour = self.screen_colour
        self.canvas = np.full(
            (self.height, self.width),
            self.screen_colour,
            dtype=np.uint8,
        )
        self.erase_active = False

        self.last_saved_path: Path | None = None  # store last saved PNG path

    def display_window(self) -> None:
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        self.window_loop()

    def window_loop(self) -> None:
        while True:
            cv2.imshow(winname=self.window_name, mat=self.canvas)
            key = cv2.waitKey(10) & 0xFF

            # Quit Draw
            if key == ord("q") or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                break

            # Clear Window
            elif key == ord("c"):
                self.reset_canvas()
                cv2.imshow(winname=self.window_name, mat=self.canvas)

            # Save drawing
            elif key == ord("s"):
                name = self._prompt_filename_dialog()
                if name:
                    self.save_arrays(name, target_size=28)

    def reset_canvas(self) -> None:
        self.canvas = np.full(
            (self.height, self.width),
            self.screen_colour,
            dtype=np.uint8,
        )

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.erase_active = (event == cv2.EVENT_RBUTTONDOWN)
            self.last_pt = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            colour = self.erase_colour if self.erase_active else self.brush_colour
            cv2.line(
                self.canvas,
                self.last_pt,
                (x, y),
                color=colour,
                thickness=self.brush_size,
                lineType=cv2.LINE_AA,
            )
            self.last_pt = (x, y)

        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            colour = self.erase_colour if self.erase_active else self.brush_colour
            cv2.line(
                self.canvas,
                self.last_pt,
                (x, y),
                color=colour,
                thickness=self.brush_size,
                lineType=cv2.LINE_AA,
            )
            self.drawing = False
            self.erase_active = False
            self.last_pt = None

    def save_csv(self, arr: np.ndarray, filename: Path) -> None:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            for row in arr:
                writer.writerow(row.tolist())

    def save_arrays(self, base_name: str = "drawing", target_size: int = 28) -> None:
        file_name = f"{base_name}_{target_size}x{target_size}"

        drawings_output_folder = Path("customFigures/original")
        drawings_output_folder.mkdir(parents=True, exist_ok=True)

        csv_output_folder = Path("csvs")
        csv_output_folder.mkdir(parents=True, exist_ok=True)

        drawing_full_file_path = drawings_output_folder / f"{file_name}.png"
        csv_full_file_path = csv_output_folder / f"{file_name}.csv"

        # Resize to target_size x target_size
        resized = cv2.resize(
            self.canvas,
            (target_size, target_size),
            interpolation=cv2.INTER_AREA,
        )

        # Save CSV and PNG
        self.save_csv(resized, csv_full_file_path)
        cv2.imwrite(str(drawing_full_file_path), resized)

        self.last_saved_path = drawing_full_file_path

        print("Saved drawing to:", drawing_full_file_path)
        print("Stats (uint8):", resized.shape, resized.min(), resized.max())

    def _prompt_filename_dialog(self, prompt="Name this drawing", default="drawing") -> str | None:
        root = tk.Tk()
        root.withdraw()
        try:
            name = simpledialog.askstring(
                title="Save Drawing",
                prompt=prompt,
                initialvalue=default,
            )
            return name
        finally:
            root.destroy()


if __name__ == "__main__":
    window = WindowDraw(width_height=128, brush_size=4)
    window.display_window()
