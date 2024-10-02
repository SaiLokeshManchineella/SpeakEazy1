import cv2
import os
import numpy as np
import mediapipe as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
from tensorflow import keras
from keras import models
from keras import layers


actions = np.array(['1. loud','2. quiet','3. happy','4. sad','5. Beautiful','6. Ugly','7. Deaf','8. Blind','9. Nice','10. Mean','11. rich','12. poor','13. thick','14. thin','15. expensive','16. cheap','17. flat','18. curved','19. male','20. female','21. tight','22. loose','23. high','24. low','25. soft','26. hard','27. deep','28. shallow','29. clean','30. dirty','31. strong','32. weak','33. dead','34. alive','35. heavy'])



mp_holistic = mp.solutions.holistic  # Holistic model
try:
    model = models.load_model("SpeakEazy\\action75a.h5")
    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for MediaPipe
        image.flags.writeable = False  # Disable writing to the image (optimization)
        results = model.process(image)  # Perform holistic model prediction
        image.flags.writeable = True  # Enable writing to the image again
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
        return image, results

    def extract_features(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, face, lh, rh])

    def extract_40_frames(video_path):
        clip = VideoFileClip(video_path)
        duration = clip.duration
        frame_indices = np.linspace(0, duration - 0.01, 40, endpoint=False)  # Calculate 40 evenly spaced frame indices
        frames = []

        for t in frame_indices:
            frame = clip.get_frame(t)
            frames.append(frame)

        clip.reader.close()
        return frames

    def process_video(video_file):
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Extract 40 frames from the video
            clip = VideoFileClip(video_file)
            duration = clip.duration
            frame_indices = np.linspace(0, duration - 0.01, 40, endpoint=False)  # Calculate 40 evenly spaced frame indices
            frames = []

            for t in frame_indices:
                frame = clip.get_frame(t)
                frames.append(frame)

            clip.reader.close()
            # Array to store landmarks for all 40 frames
            all_keypoints = []

            for idx, frame in enumerate(frames):
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_features(results)
                all_keypoints.append(keypoints)  # Append keypoints to the array
            all_keypoints = np.array(all_keypoints)
            all_keypoints  = all_keypoints .reshape((1, all_keypoints .shape[0], all_keypoints .shape[1]))
            prediction = model.predict(all_keypoints)  # Return the array of keypoints for all frames
            predicted_class_index = np.argmax(prediction[0])
            class_name = actions[predicted_class_index].split()
            return class_name[1]




except ValueError as e:
    print(f"Model loading error: {e}")


