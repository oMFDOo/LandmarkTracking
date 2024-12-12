import cv2
import mediapipe as mp
import time
import numpy as np
from datetime import datetime

# Mediapipe face mesh 설정
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 웹캠 연결
cap = cv2.VideoCapture(0)

# Face Mesh 모델 초기화
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR 이미지를 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # 랜드마크가 추적되면
        frame_with_landmarks = frame.copy()
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 랜드마크를 그릴 때 점 크기를 작게 설정
                landmark_drawing_spec = mp_drawing.DrawingSpec(color=(255, 116, 139), thickness=1, circle_radius=1)
                mp_drawing.draw_landmarks(frame_with_landmarks, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec, landmark_drawing_spec)

        # 두 이미지를 가로로 이어붙이기 (왼쪽은 원본, 오른쪽은 랜드마크 그려진 이미지)
        combined_frame = cv2.hconcat([frame, frame_with_landmarks])

        # 웹캠 영상 표시
        cv2.imshow('Webcam Feed', combined_frame)

        # 'c' 키를 눌러 캡쳐
        if cv2.waitKey(1) & 0xFF == ord('c'):
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            landmark_values = []

            # 랜드마크 값 추출
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        landmark_values.append((landmark.x, landmark.y, landmark.z))

            # 랜드마크 값 파일로 저장
            with open(f"./data/landmarks/landmarks_{timestamp}.txt", 'w') as f:
                for landmark in landmark_values:
                    f.write(f"{landmark[0]}, {landmark[1]}, {landmark[2]}\n")

            # 두 이미지(원본+랜드마크)가 결합된 상태로 저장
            cv2.imwrite(f"./data/images/combined_{timestamp}.jpg", combined_frame)

            print(f"Captured at {timestamp}")

        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
