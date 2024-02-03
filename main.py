from flask import Flask, jsonify, request
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import json

# Загрузка данных из JSON файла
with open('pro3.json', 'r') as file:
    data = json.load(file)


def create_model(data):
    global item, label_to_index, EegDataset, EegClassifier, model, batch_size
    # Подготовка данных для обучения
    X = []
    y = []
    for item in data:
        X.append([item['Attention'], item['Meditation'], item['Signal'], item['Delta'], item['Theta'],
                  item['LowAlpha'], item['HighAlpha'], item['LowBeta'], item['HighBeta'], item['LowGamma'],
                  item['HighGamma']])
        y.append(item['EventName'])
    unique_labels = list(set(y))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    # Преобразование данных в тензоры PyTorch
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_indices = [label_to_index[label] for label in y]
    y_tensor = torch.tensor(y_indices, dtype=torch.long)

    # Создание пользовательского класса Dataset
    class EegDataset(Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    dataset = EegDataset(X_tensor, y_tensor)

    # Определение архитектуры модели, функции потерь и оптимизатора
    class EegClassifier(nn.Module):
        def __init__(self, input_size, num_classes):
            super(EegClassifier, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, num_classes)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    input_size = len(X[0])
    model = EegClassifier(input_size=input_size, num_classes=len(set(y)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Обучение модели
    num_epochs = 10
    batch_size = 32
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    # Ваша модель теперь обучена и готова для использования.
    # Сохранение весов модели
    torch.save(model.state_dict(), 'trained_model.pth')
    # Сохранение модели в формате ONNX
    dummy_input = torch.randn(1, input_size)  # Создание примерного входного тензора
    torch.onnx.export(model, dummy_input, 'trained_model.onnx', input_names=['input'], output_names=['output'])
    # Предположим, что у вас уже есть обученная модель (model)

    # Пример новых данных для предсказания
    # item = {
    #     "Attention": 85,
    #     "Meditation": 60,
    #     "Signal": 80,
    #     "Delta": 50,
    #     "Theta": 30,
    #     "LowAlpha": 25,
    #     "HighAlpha": 35,
    #     "LowBeta": 40,
    #     "HighBeta": 55,
    #     "LowGamma": 20,
    #     "HighGamma": 15
    # }
    # # Преобразование новых данных в тензор PyTorch
    # new_data_tensor = torch.tensor(
    #     [[item['Attention'], item['Meditation'], item['Signal'], item['Delta'], item['Theta'],
    #       item['LowAlpha'], item['HighAlpha'], item['LowBeta'], item['HighBeta'],
    #       item['LowGamma'], item['HighGamma']]], dtype=torch.float32)
    # # Установка модели в режим предсказания (evaluation mode)
    # model.eval()
    # # Предсказание на новых данных
    # with torch.no_grad():
    #     outputs = model(new_data_tensor)
    #     _, predicted_indices = torch.max(outputs, 1)
    # # Преобразование числовых индексов обратно в метки классов
    # index_to_label = {index: label for label, index in label_to_index.items()}
    # predicted_labels = [index_to_label[idx.item()] for idx in predicted_indices]
    # print("Predicted EventNames:")
    # for label in predicted_labels:
    #     print(label)


create_model(data)

# Создание Flask-приложения
app = Flask(__name__)

# Эндпоинт для получения предсказаний на основе данных
@app.route('/get', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        # for postman test
        # item = request.json
        # Предполагая, что данные поступают в формате JSON
        gitem = json.loads(request.json)
        print(item)

        # Преобразование новых данных в тензор PyTorch
        new_data_tensor = torch.tensor(
            [[item['Attention'], item['Meditation'], item['Signal'], item['Delta'], item['Theta'],
              item['LowAlpha'], item['HighAlpha'], item['LowBeta'], item['HighBeta'],
              item['LowGamma'], item['HighGamma']]], dtype=torch.float32)

        # Установка модели в режим предсказания (evaluation mode)
        model.eval()

        # Предсказание на новых данных
        with torch.no_grad():
            outputs = model(new_data_tensor)
            _, predicted_indices = torch.max(outputs, 1)

        # Преобразование числовых индексов обратно в метки классов
        index_to_label = {index: label for label, index in label_to_index.items()}
        predicted_labels = [index_to_label[idx.item()] for idx in predicted_indices]

        print("Predicted EventNames:")
        for label in predicted_labels:
            print(label)

        return jsonify({'EventName': label})

# Эндпоинт для обновления модели на основе данных
@app.route('/set', methods=['POST'])
def update_model():
    if request.method == 'POST':
        models_data = json.loads(request.json)  # Получаем данные о моделях из POST-запроса
        create_model(models_data)

        return "Модель обновлена успешно"

if __name__ == '__main__':
    app.run(debug=False)