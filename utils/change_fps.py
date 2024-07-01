import cv2

def change_fps(
        frames_processed, 
        frames_queued, 
        frame_fps, 
        frame_width, 
        frame_height, 
        video_in_path, 
        video_out_path
        ):
    target_fps = int(round((frames_processed*1.0/frames_queued) * frame_fps))
    print(f"Target FPS: {target_fps}")

    # Initialize video writer and reader
    changed_fps_video_writer = cv2.VideoWriter(
        video_out_path, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        target_fps, 
        (frame_width, frame_height)
    )
    change_fps_video_capture = cv2.VideoCapture(video_in_path)

    # Write frames
    while change_fps_video_capture.isOpened():
        ret, frame = change_fps_video_capture.read()
        if ret:
            changed_fps_video_writer.write(frame)
        else:
            break

    # Release reader and writer
    changed_fps_video_writer.release()
    change_fps_video_capture.release()