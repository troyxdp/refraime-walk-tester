import cv2, queue, threading

# bufferless VideoCapture
class RTSPVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        self.frame_count = 0
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

# read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            self.frame_count += 1
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)
        return

    def read(self):
        return self.q.get()
    
    def release(self):
        self.cap.release()