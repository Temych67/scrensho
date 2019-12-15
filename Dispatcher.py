from screenshot import doing_screenshot
import dlib
from skimage import io
from  scipy.spatial import distance

doing_screenshot()

sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

# Загружаем первую фотографию
img = io.imread('001.jpg')

# Показываем фотографию средствами dlib
win1 = dlib.image_window()
win1.clear_overlay()
win1.set_image(img)



# Находим лицо на фотографии
dets = detector(img, 1)

for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    shape = sp(img, d)
    win1.clear_overlay()
    win1.add_overlay(d)
    win1.add_overlay(shape)

# Извлекаем дескриптор из лица
face_descriptor1 = facerec.compute_face_descriptor(img, shape)


# Загружаем и обрабатываем вторую фотографию сделаную из скрина
img1 = io.imread('camer.png')
win2 = dlib.image_window()
win2.clear_overlay()
win2.set_image(img1)
dets_webcam = detector(img1, 1)
for k, d in enumerate(dets_webcam):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    shape = sp(img1, d)
    win2.clear_overlay()
    win2.add_overlay(d)
    win2.add_overlay(shape)

# Извлекаем дескриптор из лица
face_descriptor2 = facerec.compute_face_descriptor(img1, shape)

a = distance.euclidean(face_descriptor1, face_descriptor2)
if a <= 0.6:
    print("Identity is confirmed")
else:
    print("Identity is not confirmed")

