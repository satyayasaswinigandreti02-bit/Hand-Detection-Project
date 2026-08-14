import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)

timestamp = 0

while True:
    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp += 1

    results = landmarker.detect_for_video(mp_image, timestamp)

    if results.hand_landmarks:
        hand_count = len(results.hand_landmarks)

        if hand_count == 2:
            display_text = "BOTH HANDS"
        else:
            hand_name = results.handedness[0][0].category_name
            display_text = hand_name.upper() + " HAND"

        print(display_text)

        for hand_landmarks in results.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        cv2.putText(
            frame,
            display_text,
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )
    else:
        display_text = "NO HAND DETECTED"

        cv2.putText(
            frame,
            display_text,
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
landmarker.close()
cv2.destroyAllWindows()