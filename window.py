import csv
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from pathlib import Path

class WindowDraw:
    def __init__(self, name : str = "Draw", width_height : int = 128, brush_size : int = 4) -> None:
        self.width, self.height = width_height, width_height
        self.window_name = name
        self.drawing = False
        self.last_pt = None
        self.brush_size = brush_size
        self.min_brush_size = 0
        self.max_brush_size = 8
        self.brush_colour = 0xFF
        self.screen_colour = self.brush_colour ^ 0xFF
        self.erase_colour = self.screen_colour
        self.canvas = np.full((self.height, self.width), self.screen_colour, dtype=np.uint8)
        self.erase_active = False

    def display_window(self):
        cv2.namedWindow(self.window_name)
        # cv2.createTrackbar("Brush", self.window_name, self.brush_size, self.max_brush_size, self.on_trackbar)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        self.window_loop()

    def window_loop(self):
        while True:
            cv2.imshow(winname=self.window_name, mat=self.canvas)
            key = cv2.waitKey(10) & 0xFF
            # Quit Draw
            if key == ord('q') or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                break
             # Clear Window
            elif key == ord('c'):
                self.reset_canvas()
                cv2.imshow(winname=self.window_name, mat=self.canvas)
            elif key == ord('s'):
                name = self._prompt_filename_dialog()
                if name:
                    self.save_arrays(name, target_size=28)
                    cv2.destroyAllWindows()
                    break

    def reset_canvas(self):
        self.canvas = np.full((self.height, self.width), self.screen_colour ,dtype=np.uint8)
    
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.erase_active = (event == cv2.EVENT_RBUTTONDOWN)
            self.last_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Draw a smooth stroke between points
            colour = self.erase_colour if self.erase_active else self.brush_colour
            cv2.line(self.canvas, self.last_pt, (x, y), color=colour, thickness=self.brush_size, lineType=cv2.LINE_AA)
            self.last_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            # Final segment + stop drawing
            colour = self.erase_colour if self.erase_active else self.brush_colour
            cv2.line(self.canvas, self.last_pt, (x, y), color=colour, thickness=self.brush_size, lineType=cv2.LINE_AA)
            self.drawing = False
            self.erase_active = False
            self.last_pt = None

    # def on_trackbar(self, val):
    #     # Keep within limits
    #     v = max(self.min_brush_size, min(self.max_brush_size, val))
    #     v = v if v % 2 == 1 else v + 1
    #     cv2.setTrackbarPos("Brush", self.window_name, v)
    #     self.brush_size = v
        
    def save_csv (self, arr: np.ndarray, filename : str):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in arr:
                writer.writerow(row.tolist())
        
    def save_arrays(self, base_name: str = "drawing", target_size: int = 28):
        file_name = f"{base_name}_{target_size}x{target_size}"
        
        drawings_output_folder = Path("drawings")
        drawings_output_folder.mkdir(parents=True, exist_ok=True)
        csv_output_folder = Path("csvs")
        csv_output_folder.mkdir(parents=True, exist_ok=True)
        drawing_full_file_path = drawings_output_folder / file_name
        csv_full_file_path = csv_output_folder / file_name
        
        # Make a target_size x target_size version, INTER_AREA is the default; you can use INTER_NEAREST if you want more blocky pixels.
        resized = cv2.resize(self.canvas, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # Save the array data in a .npy file
        # np.save(f"{file_name}.npy", resized)  # uint8
        
        # Additionally, save the data as .csv
        self.save_csv(resized, f"{csv_full_file_path}.csv")

        # Save a PNG preview at the target size
        cv2.imwrite(f"{drawing_full_file_path}.png", resized)

        print("Stats (uint8):", resized.shape, resized.min(), resized.max())

    def _prompt_filename_dialog(self, prompt="Name this drawing", default="drawing"):
        root = tk.Tk()
        root.withdraw()  # hide main window
        try:
            name = simpledialog.askstring(title="Save Drawing", prompt=prompt, initialvalue=default)
            return name
        finally:
            root.destroy()

if __name__ == "__main__":
    window = WindowDraw(brush_size=4)
    window.display_window()
    print(window.canvas)