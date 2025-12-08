# runner.py
"""
End-to-end runner:
1. Open drawing window.
2. Let the user draw and press 's' to save a 28x28 digit.
3. After the window is closed / quit (press q), run the CNN on the last saved drawing and print prediction.
"""

from pathlib import Path

from window import WindowDraw
from predict_custom import load_model, predict_image_path


def main():
    # 1. Open drawing window and let the user interact
    window = WindowDraw(width_height=128, brush_size=4)
    window.display_window()  # blocks until window is closed

    # 2. After closing, check what was last saved
    if window.last_saved_path is None:
        print("No drawing was saved this session.")
        return

    img_path: Path = window.last_saved_path
    print(f"Using last saved drawing: {img_path}")

    # 3. Load model and run prediction
    model = load_model()
    pred_class, top_probs, top_idxs = predict_image_path(img_path, model)

    print(f"Predicted digit: {pred_class}")
    print("Top-3 predictions:")
    for p, idx in zip(top_probs, top_idxs):
        print(f"  {idx.item()}: {p.item():.4f}")


if __name__ == "__main__":
    main()
