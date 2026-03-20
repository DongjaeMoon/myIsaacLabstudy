import cv2
from pathlib import Path

out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

tag_id = 0
side_pixels = 800

img = dictionary.generateImageMarker(tag_id, side_pixels)
out_path = out_dir / f"tag36h11_id{tag_id}.png"
cv2.imwrite(str(out_path), img)

print(f"saved: {out_path}")
