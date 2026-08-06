import threading
import time
from typing import Optional

import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT
)

from core.console import log


class Camera:

    def __init__(self):

        self.capture: Optional[cv2.VideoCapture] = None

        self.frame: Optional[np.ndarray] = None

        self.running = False

        self.thread: Optional[threading.Thread] = None



    def start(self):

        if self.running:
            return


        log("CAMERA", "Starting...")


        capture = cv2.VideoCapture(
            CAMERA_INDEX
        )


        capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH
        )

        capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT
        )


        if not capture.isOpened():

            log(
                "CAMERA",
                "Failed to open webcam"
            )

            capture.release()

            return


        self.capture = capture

        self.running = True


        self.thread = threading.Thread(
            target=self._loop,
            daemon=True
        )

        self.thread.start()


        log("CAMERA", "Ready")



    def _loop(self):

        while self.running:

            if self.capture is None:
                break


            success, frame = self.capture.read()


            if success:

                self.frame = frame


            time.sleep(
                0.03
            )



    def get_frame(
        self
    ) -> Optional[np.ndarray]:

        return self.frame



    def stop(self):

        self.running = False


        if self.capture:

            self.capture.release()


        self.capture = None



camera = Camera()