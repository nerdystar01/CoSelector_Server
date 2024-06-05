import os, gc, datetime, subprocess, pickle, argparse, sys, shutil
import numpy as np
from PIL import Image, ImageFile
import requests
from io import BytesIO
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import ResNet50

# 최초 인증 버전인 241009 버전의 개량형. 경량 모델 ( 확률 예상치 65-70% )
# 배포버전 : 240527

# 설정 파일 직접 입력
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
batch_size = 80
EPOCHS = 10
input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)  # 이를 함수 내부에 정의하거나 전역 변수로 설정

print("TensorFlow 버전:", tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
if not physical_devices:
    print("사용 가능한 GPU가 없습니다. Window 기준 Tensorflow 2.9 필요")
    sys.exit(1)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 학습 데이터 준비 클래스
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, positive_data_folder, negative_data_folder, batch_size=32, is_validation=False):
        self.positive_data_folder = positive_data_folder
        self.negative_data_folder = negative_data_folder
        self.batch_size = batch_size
        self.positive_data = self.load_data(self.positive_data_folder)
        self.negative_data = self.load_data(self.negative_data_folder)
        split_idx = int(len(self.positive_data) * 0.8)
        if is_validation:
            self.positive_data = self.positive_data[split_idx:]
            self.negative_data = self.negative_data[split_idx:]
    
    def load_data(self, folder_path):
        img_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith('.png') or fname.endswith('.jpg')]
        return img_paths

    def __len__(self):
        return len(self.positive_data)

    def __getitem__(self, index):
        X1_batch = []
        X2_batch = []
        y_batch = []
        pos_img = np.array(Image.open(self.positive_data[index]))
        for _ in range(self.batch_size // 2):
            neg_index = np.random.choice(len(self.negative_data))
            X1_batch.append(pos_img)
            X2_batch.append(np.array(Image.open(self.negative_data[neg_index])))
            y_batch.append(0)
        for _ in range(self.batch_size // 2):
            while True:
                pos_index2 = np.random.choice(len(self.positive_data))
                if pos_index2 != index:
                    break
            X1_batch.append(pos_img)
            X2_batch.append(np.array(Image.open(self.positive_data[pos_index2])))
            y_batch.append(1)
        return [np.array(X1_batch, dtype='float32'), np.array(X2_batch, dtype='float32')], np.array(y_batch, dtype='int8')

# LOSS 함수 (Embedding Vector용)
def contrastive_loss(y_true, y_pred, margin=1.0):

    y_true = tf.cast(y_true, tf.float32)
    # 두 임베딩의 거리 계산
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - y_pred, 0))
    # Contrastive 손실 계산
    return tf.math.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# Resize_image, 학습된 모델의 사이즈와 동일해야함. 
def resize_image(image, target_width=IMAGE_WIDTH, target_height=IMAGE_HEIGHT):
    # Aspect ratio와 목표 차원을 기반으로 새로운 차원을 계산
    aspect_ratio = image.width / image.height
    if image.width > image.height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    image = image.resize((new_width, new_height), Image.LANCZOS)
    # 새 이미지 객체 생성 및 중앙에 원본 이미지 붙여넣기
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    new_image.paste(image, ((target_width - new_width) // 2, (target_height - new_height) // 2))
    return np.array(new_image, dtype='float32')

def get_embedding_filename(model_name):
    # 모델 이름에서 직접 파일 경로 생성
    keras_path = os.path.join(BASE_DIR, 'tastemodels', f'{model_name}.keras')
    pkl_path = os.path.join(BASE_DIR, 'tastemodels', f'{model_name}.pkl')
    return keras_path, pkl_path

# 임베딩을 활용한 벡터 추출
def compute_embeddings(model, positive_images, model_name):
    print("\n긍정 이미지의 Embedding Vector 값을 계산합니다\n")
    embeddings = []
    num_images = len(positive_images)
    for idx, img_path in enumerate(positive_images):
        image = np.array([resize_image(Image.open(img_path))], dtype='float32')
        embedding_outputs = model.predict([image, image])  # Assuming the model expects two inputs
        embedding = embedding_outputs[0][0]
        embeddings.append(embedding)
        print(f"\rProcessing image {idx + 1} out of {num_images}", end="")
    print("\nEmbedding Vector 값이 계산되어 저장되었습니다.")
    
    _, pkl_path = get_embedding_filename(model_name)
    # Pickle을 사용하여 임베딩 벡터 리스트 저장
    with open(pkl_path, 'wb') as f:
        pickle.dump(embeddings, f)

# 유사도 평가 ( 파일로 저장하지 않고 PICK할때 인스턴스하게 확인 )
def evaluate_similarity(new_embedding, embeddings, num_samples=12, exclude_top=2, num_avg=5):
    def compute_random_avg_embedding(embeddings):
        indices = np.arange(len(embeddings))  # 임베딩 리스트의 인덱스 배열 생성
        selected_indices = np.random.choice(indices, num_samples, replace=False)
        selected_embeddings = [embeddings[i] for i in selected_indices]
        distances = [np.linalg.norm(e - np.mean(selected_embeddings, axis=0)) for e in selected_embeddings]
        sorted_indices = np.argsort(distances)
        top_embeddings = [selected_embeddings[i] for i in sorted_indices[:-exclude_top]]
        avg_embedding = np.mean(top_embeddings, axis=0)
        return avg_embedding
    avg_embeddings = [compute_random_avg_embedding(embeddings) for _ in range(num_avg)]
    distances = [np.linalg.norm(new_embedding - avg_embedding) for avg_embedding in avg_embeddings]
    return min(distances)

def get_embedding_model(model, input_shape):
    # 주어진 샴 네트워크에서 이미 정의된 기본 네트워크(base_network)를 가져옵니다.
    base_network = model.layers[2]
    # 새로운 입력 레이어 생성
    input_layer = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    # 임베딩 계산및 모델 반환
    embeddings = base_network(input_layer)
    return tf.keras.models.Model(inputs=input_layer, outputs=embeddings)

#ResNet-50 모델을 사용하여 추가 레이어
def Train_HolyGrail(input_shape):
    # 사전 학습된 ResNet-50 모델 불러오기
    def create_base_model():
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        # ResNet-50 모델의 레이어를 동결 (추가 학습을 위해 필요한 경우 변경 가능)
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        embedding_output = tf.keras.layers.Dense(128)(x)  # 활성화 함수 없음
        return tf.keras.models.Model(inputs=base_model.input, outputs=embedding_output)
    
    # 두 이미지를 입력으로 받을 수 있도록 입력 레이어를 정의합니다.
    input_image1 = tf.keras.layers.Input(input_shape)
    input_image2 = tf.keras.layers.Input(input_shape)
    
    base_network = create_base_model()
    
    embedding_1 = base_network(input_image1)
    embedding_2 = base_network(input_image2)
    
    siamese_network = tf.keras.models.Model(inputs=[input_image1, input_image2], outputs=[embedding_1, embedding_2])
    return siamese_network

def process_pick(model_name, target_folder):
    model_path = os.path.join(BASE_DIR, 'tastemodels', f'{model_name}.keras')
    pkl_path = os.path.join(BASE_DIR, 'tastemodels', f'{model_name}.pkl')
    model = tf.keras.models.load_model(model_path, custom_objects={"contrastive_loss": contrastive_loss})
    embedding_model = get_embedding_model(model, input_shape)
    
    with open(pkl_path, 'rb') as f:
        embeddings = pickle.load(f)

    candidate_images = [os.path.join(target_folder, f) for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))]
    total_files = len(candidate_images)
    pick_by_ai_folder = os.path.join(target_folder, "PickByAi")
    os.makedirs(pick_by_ai_folder, exist_ok=True)
    picked_images = []

    for idx, img_path in enumerate(candidate_images):
        print(f"Processing file {idx + 1} out of {total_files}...", end='\r')
        image = resize_image(Image.open(img_path))
        image_np = np.array([image], dtype='float32')
        new_embedding = embedding_model.predict(image_np)[0]
        distance = evaluate_similarity(new_embedding, embeddings)

        # 거리값을 백분율로 변환 (0-50 사이)
        if distance >= 50:
            similarity_percentage = 0.0
        else:
            similarity_percentage = (50 - distance) / 50 * 100

        similarity_score = f"{similarity_percentage:.2f}%"
        new_file_name = f"{similarity_score}_{os.path.basename(img_path)}"
        shutil.copy(img_path, os.path.join(pick_by_ai_folder, new_file_name))
        picked_images.append((new_file_name, similarity_score))

    # Save the results to a text file
    results_filename = os.path.join(pick_by_ai_folder, "picked_images_summary.txt")
    with open(results_filename, 'w', encoding='utf-8') as f:  # 인코딩을 utf-8로 지정
        for filename, score in picked_images:
            f.write(f"{filename}: {score}\n")

    return len(candidate_images)  # 선택된 이미지의 수 반환

def process_pick_with_api(model_name, resource_list):
    # Load evaluation model
    model_path = os.path.join(BASE_DIR, 'tastemodels', f'{model_name}.keras')
    pkl_path = os.path.join(BASE_DIR, 'tastemodels', f'{model_name}.pkl')
    model = tf.keras.models.load_model(model_path, custom_objects={"contrastive_loss": contrastive_loss})
    embedding_model = get_embedding_model(model, input_shape)
    
    with open(pkl_path, 'rb') as f:
        embeddings = pickle.load(f)
    
    for resource_object in resource_list:
        image_url = resource_object['image_url']
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # 에러가 발생하면 예외 발생
            image = Image.open(BytesIO(response.content))
            if image is not None:
                image_np = resize_image(image)
                new_embedding = embedding_model.predict(image_np)[0]
                distance = evaluate_similarity(new_embedding, embeddings)

                if distance >= 50:
                    similarity_percentage = 0.0
                else:
                    similarity_percentage = (50 - distance) / 50 * 100
                    similarity_score = f"{similarity_percentage:.2f}%"
                
                resource_object['similarity_score'] = similarity_score
            else:
                # Handle the case where image cannot be retrieved
                resource_object['similarity_score'] = "Image not available"
        except Exception as e:
            # Handle the exception if image cannot be retrieved or processed
            resource_object['similarity_score'] = f"Error: {str(e)}"

    return resource_list


# TUNE 모드 함수 ( 오버 피팅 가능성이 높아서 Embedding 방식에서 Fine-Tune은 조금더 검증해야 함)
def finetune(target_folder, model, batch_size_finetune=16, learning_rate=0.0001):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=contrastive_loss)
    
    # 이미지 불러오기
    success_images = [os.path.join(target_folder, 'success', f) for f in os.listdir(os.path.join(target_folder, 'success'))]
    fail_images = [os.path.join(target_folder, 'fail', f) for f in os.listdir(os.path.join(target_folder, 'fail'))]

    for pos_img_path in success_images:
        pos_img = resize_image(Image.open(pos_img_path))

        # 긍정-긍정 셋 구성
        positive_samples = np.random.choice([img for img in success_images if img != pos_img_path], batch_size_finetune // 2, replace=False)
        X1_pos = np.array([pos_img] * (batch_size_finetune // 2))
        X2_pos = np.array([resize_image(Image.open(img_path)) for img_path in positive_samples])

        # 긍정-부정 셋 구성
        negative_samples = np.random.choice(fail_images, batch_size_finetune // 2, replace=False)
        X1_neg = np.array([pos_img] * (batch_size_finetune // 2))
        X2_neg = np.array([resize_image(Image.open(img_path)) for img_path in negative_samples])

        X1_batch = np.concatenate([X1_pos, X1_neg], axis=0)
        X2_batch = np.concatenate([X2_pos, X2_neg], axis=0)
        y_batch = np.concatenate([np.ones(batch_size_finetune // 2), np.zeros(batch_size_finetune // 2)], axis=0)

        model.train_on_batch([X1_batch, X2_batch], y_batch)

    return model

def main():
    # 명령행 인자 처리
    parser = argparse.ArgumentParser(description='모델 학습, 선택, 미세조정 모드')
    parser.add_argument('-t', '--train', type=str, help='학습 모드에서 사용할 데이터 폴더 이름')
    parser.add_argument('-p', '--pick', nargs=2, help='PICK 모드에서 사용할 모델 이름과 대상 폴더 이름')
    parser.add_argument('-f', '--finetune', action='store_true', help='미세조정 모드')
    args = parser.parse_args()

    start_time = datetime.datetime.now()
    print(f"작업 시작 시각: {start_time}")

    if args.train:
        print("Train mode activated.")
        train_data_folder = args.train
        train_data_dir = os.path.join(BASE_DIR, train_data_folder)
        positive_data_folder = os.path.join(train_data_dir, 'PO')
        negative_data_folder = os.path.join(train_data_dir, 'NG')
        model_name = train_data_folder  # 모델 이름을 훈련 폴더 이름으로 설정

        model = Train_HolyGrail(input_shape)
        model.compile(optimizer='adam', loss=contrastive_loss)
        train_gen = DataGenerator(positive_data_folder, negative_data_folder, batch_size, is_validation=False)
        val_gen = DataGenerator(positive_data_folder, negative_data_folder, batch_size, is_validation=True)
        model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
        # 모델 및 임베딩 파일 저장 경로 설정
        keras_path, _ = get_embedding_filename(model_name)
        model.save(keras_path)
        positive_images = [os.path.join(positive_data_folder, f) for f in os.listdir(positive_data_folder) if f.endswith('.png') or f.endswith('.jpg')]
        compute_embeddings(model, positive_images, model_name)

    elif args.pick:
        model_name, pick_folder = args.pick
        print(f"PICK mode activated. Model: {model_name}, Target folder: {pick_folder}")
        num_images = process_pick(model_name, pick_folder)
        print(f"{pick_folder} 폴더에서 이미지를 총 {num_images} 개 찾았습니다.")
        subprocess.Popen(f'explorer {os.path.realpath(os.path.join(pick_folder, "PickByAi"))}')

    elif args.finetune:
        print("Fine-tune mode activated.")
        model = tf.keras.models.load_model(MODEL_FILE_PATH, custom_objects={"contrastive_loss": contrastive_loss})
        target_folder = os.path.join(BASE_DIR, args.directory)  # 사용자로부터 입력 받은 폴더 경로
        for target_folder in target_folders:
            model = finetune(target_folder, model)
        model.save(MODEL_FILE_PATH)

    end_time = datetime.datetime.now()
    print(f"작업 종료 시각: {end_time}")

if __name__ == "__main__":
    main()