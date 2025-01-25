from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO


class Tracker():
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)
        self.line = [(0, 300), (640, 300)] # default line
        self.track_history = defaultdict(lambda: [])
        self.num_enter = 0
        self.num_exit = 0
    
    def track(self, video_path) -> list:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Error opening video file.")


        frames = []

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                try:
                    h, w, _ = frame.shape
                    self.line = [(0, h // 2), (w, h // 2)]

                    results = self.model.track(frame, persist=True, verbose=True)

                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                    annotated_frame = results[0].plot()

                    cv2.line(annotated_frame, self.line[0], self.line[1], (0, 255, 0), 2)

                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        cx = x + w / 2
                        cy = y + h / 2

                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                        track = self.track_history[track_id]
                        track.append((float(cx), float(cy)))
                        if len(track) > 30:
                            track.pop(0)

                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                        if len(track) > 1:
                            last_x, last_y = track[-2]
                            dy = cy - last_y
                            if dy < 0:
                                direction = "Exit"
                            elif dy > 0:
                                direction = "Enter"

                            if (last_y > self.line[0][1] and cy <= self.line[0][1]) or (last_y < self.line[0][1] and cy >= self.line[0][1]):
                                cv2.line(annotated_frame, self.line[0], self.line[1], (255, 0, 255), 2)
                                if direction == "Enter":
                                    self.num_enter += 1
                                elif direction == "Exit":
                                    self.num_exit += 1

                    cv2.putText(annotated_frame, f"Enter: {self.num_enter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(annotated_frame, f"Exit: {self.num_exit}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    frames.append(annotated_frame)

                except Exception as e:
                    print(f"Exception occurred while processing a frame: {e}")
                    continue

        except Exception as e:
            print(f"Exception occurred during video processing: {e}")

        finally:
            self.num_enter = 0
            self.num_exit = 0
            self.track_history.clear()
            cap.release()

        return frames

    def convert_video(self, frames, output_path):
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        print("Writing frame to video...")
        for frame in frames:
            try:
                out.write(frame)
            except Exception as e:
                print(f"Error writing frame to video: {e}")
                continue
        
        out.release()



if __name__ == "__main__":
    tracker = Tracker("yolov8n.pt")
    frames = tracker.track("test3.mp4")
    for frame in frames:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()