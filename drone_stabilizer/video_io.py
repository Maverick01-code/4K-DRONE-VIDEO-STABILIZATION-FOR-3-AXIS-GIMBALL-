from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Tuple


class VideoReader:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    def frames(self) -> Generator[np.ndarray, None, None]:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def seek(self, frame_idx: int) -> None:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __len__(self) -> int:
        return self.frame_count


class VideoWriter:
    def __init__(
        self,
        path: str | Path,
        fps: float,
        size: Tuple[int, int],
        codec: str = "mp4v",
    ) -> None:
        self.path = str(path)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(self.path, fourcc, fps, size)
        if not self._writer.isOpened():
            raise IOError(f"Cannot open VideoWriter at: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()
