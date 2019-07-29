import os
import time

import dlib
import cv2

from facealigner import FaceAligner


data_path = "/Users/yuhua.cheng/Documents/dataset/ethnicity_v1/0"
cnn_face_detector_model_path = '/Users/yuhua.cheng/PycharmProjects/model/mmod_human_face_detector.dat'
five_landmarks_model_path = '/Users/yuhua.cheng/PycharmProjects/model/shape_predictor_5_face_landmarks.dat'
sixty_eight_landmarks_model_path = '/Users/yuhua.cheng/PycharmProjects/model/shape_predictor_68_face_landmarks.dat'
face_embedding_model_path = '/Users/yuhua.cheng/PycharmProjects/model/dlib_face_recognition_resnet_model_v1.dat'

def detect_with_hog(detector, img, height=512, width=0):
	w, h, c = img.shape
	if not width:
		width = int((height/h)*w)
	img = cv2.resize(img, (height, width))
	t0 = time.time()
	faceRects = detector(img, 0)
	t1 = time.time()
	if len(faceRects) == 0:
		faceRects = detector(img, 2)
	for faceRect in faceRects:
		cv2.rectangle(img, (faceRect.left(), faceRect.top()), (faceRect.right(), faceRect.bottom()), (0, 225, 0), 4, 4)
	cv2.imshow("hog_detector", img)
	cv2.setWindowTitle("hog_detector", "time:{}".format(t1 - t0))


def detect_with_cnn(detector, img, height=512, width=0):
	w, h, c = img.shape
	if not width:
		width = int((height / h) * w)
	img = cv2.resize(img, (height, width))
	t0 = time.time()
	faceRects = detector(img, 0)
	t1 = time.time()
	if len(faceRects) == 0:
		faceRects = detector(img, 2)
	for faceRect in faceRects:
		cv2.rectangle(img, (faceRect.rect.left(), faceRect.rect.top()), (faceRect.rect.right(), faceRect.rect.bottom()), (0, 255, 0), 4, 4)
	cv2.imshow("cnn_detector", img)
	cv2.setWindowTitle("cnn_detector", "time:{}".format(t1 - t0))

def align_face(detector, five_landmarks_predictor, rgb_img, height=512, width=0):
	gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
	faceRects = detector(rgb_img, 0)
	faces = dlib.full_object_detections()
	for faceRect in faceRects:
		faces.append(five_landmarks_predictor(rgb_img, faceRect))

	if len(faces) != 0:
		face = dlib.get_face_chip(rgb_img, faces[0], size=224, padding=0.25)
		cv2.imshow("aligned_Face", face)

def show_landmarks(detector, landmarks_predictor, rgb_img, height=512, width=0):
	faceRects = detector(rgb_img, 0)
	faces = dlib.full_object_detections()
	for faceRect in faceRects:
		shape = landmarks_predictor(rgb_img, faceRect)
		for i in range(shape.num_parts):
			coords = shape.part(i).x, shape.part(i).y
			cv2.circle(rgb_img, coords, 2, (255, 0, 0), 2)
	cv2.imshow("landmarks", rgb_img.copy())
	if shape is not None:
		return shape

def face_embedding(detector, landmarks_predictor, embedding_predictor, rgb_img, height=512, width=0):
	faceRects = detector(rgb_img, 0)
	faces = dlib.full_object_detections()
	for faceRect in faceRects:
		shape = landmarks_predictor(rgb_img, faceRect)
		face_embedding = embedding_predictor.compute_face_descriptor(rgb_img, shape)
		print("face embedding shape:", face_embedding.shape)


if __name__ == "__main__":
	hog_detector = dlib.get_frontal_face_detector()
	cnn_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector_model_path)
	five_landmarks_predictor = dlib.shape_predictor(five_landmarks_model_path)
	sixty_eight_landmarks_predictor = dlib.shape_predictor(sixty_eight_landmarks_model_path)
	embedding_predictor = dlib.face_recognition_model_v1(face_embedding_model_path)

	imgs = [f for f in os.listdir(data_path) if not f.startswith('.')]
	for _ in imgs:
		bgr_img = cv2.imread(os.path.join(data_path, _), 1)
		rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
		# detect_with_hog(hog_detector, rgb_img, 300)
		# detect_with_cnn(cnn_d       etector, rgb_img, 300)
		align_face(hog_detector, five_landmarks_predictor, rgb_img, 300)
		# show_landmarks(hog_detector, five_eight_landmarks_predictor, rgb_img, 300)
		shape = show_landmarks(hog_detector, sixty_eight_landmarks_predictor, bgr_img, 300)
		aligner = FaceAligner(landmarks=shape)
		aligned_face = aligner.align(rgb_img)
		cv2.imshow("68_landmarks_aligned_face", aligned_face)
		face_embedding(hog_detector, sixty_eight_landmarks_predictor, embedding_predictor, rgb_img, 300)
		cv2.waitKey(0)