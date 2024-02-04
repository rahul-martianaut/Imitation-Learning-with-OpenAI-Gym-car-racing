import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip, Normalize
from Data_loader import CarRacingDataset, CarRacingDataset_RNN
from model import CNNClassifier, CNN_RNN_Classifier
from torch.utils.data import DataLoader
from misc import visualize
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.metrics import precision_score, recall_score, f1_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# TRAINING

def train(model, dataloader, loss_fn, optimizer):
    model.train()
    training_loss = 0
    training_acc = 0
    run_loss = 0
    total_train = 0
    correct_train = 0
    loop = tqdm(dataloader, leave=True)

    for batch_idx, (input, label) in enumerate(loop):
        input = input.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        output = model(input)  # input one batch to model

        loss = loss_fn(output.view(-1, output.size(2)),
                       label.argmax(dim=2).view(-1).long())  # Calculates average loss of batch

        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

        run_loss += loss.item() * input.size(0)
        _, predicted = torch.max(output.data, 2)
        total_train += label.view(-1, label.size(2)).size(0)
        correct_train += (predicted == label.argmax(dim=2)).sum().item()

    training_acc += (100 * correct_train / total_train)
    training_loss += (run_loss / len(dataloader))  # mean loss for all batches

    return training_loss, training_acc


def validation(model, val_loader, loss_fn):  # VALIDATION
    model.eval()  # evaluation mode
    validation_loss = 0
    validation_acc = 0
    total_test = 0
    correct_test = 0
    val_loss = 0

    with torch.no_grad():
        val_loop = tqdm(val_loader, leave=True)
        for batch_idx, (val_input, val_label) in enumerate(val_loop):
            val_input, val_label = val_input.to(DEVICE), val_label.to(DEVICE)

            val_output = model(val_input)
            loss = loss_fn(val_output.view(-1, val_output.size(2)), val_label.argmax(dim=2).view(-1).long())
            val_loss += loss.item() * val_input.size(0)
            _, predicted = torch.max(val_output.data, 2)
            total_test += val_label.view(-1, val_label.size(2)).size(0)
            correct_test += (predicted == val_label.argmax(dim=2)).sum().item()

    validation_acc += (100 * correct_test / total_test)
    validation_loss += (val_loss / len(val_loader))

    return validation_loss, validation_acc


def test(model, test_loader, loss_fn):
    model.eval()
    correct_test = 0
    total_test = 0
    total_loss = 0
    test_loss = 0

    with torch.no_grad():
        for input, label in test_loader:
            input, label = input.to(DEVICE), label.to(DEVICE)
            output = model(input)
            loss = loss_fn(output.view(-1, output.size(2)), label.argmax(dim=2).view(-1).long())
            test_loss += loss.item() * input.size(0)
            _, predicted = torch.max(output.data, 2)
            total_test += label.view(-1, label.size(2)).size(0)
            correct_test += (predicted == label.argmax(dim=2)).sum().item()

            # predicted_flat = predicted.view(-1).cpu().numpy()
            # label_flat = label.argmax(dim=2).view(-1).cpu().numpy()
            #
            # precision = precision_score(label_flat, predicted_flat, average='weighted')
            # recall = recall_score(label_flat, predicted_flat, average='weighted')
            # f1 = f1_score(label_flat, predicted_flat, average='weighted')
            #
            # print("Precision: {:.4f}".format(precision))
            # print("Recall: {:.4f}".format(recall))
            # print("F1 Score: {:.4f}".format(f1))

    accuracy = (100 * correct_test / total_test)
    total_loss += test_loss / len(test_loader)

    return accuracy, total_loss


def main():
    model = CNN_RNN_Classifier(in_channels=3, out_size=4).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    BATCH_SIZE = 32
    EPOCH = 15

    TRAIN_4 = False
    TEST_4 = False
    PLOT = False

    data = np.load("training_data_4class_new.npz")
    images = data['images']
    labels = data['labels'].astype(float)

    # Train data
    split_ratio = 0.6
    split_index = int(len(images) * split_ratio)

    # 4 class
    X_train, X_temp = images[:split_index], images[split_index:]
    y_train, y_temp = labels[:split_index], labels[split_index:]

    # Validation, Test data
    split_ratio_val_test = 0.5
    split_index_val = int(len(X_temp) * split_ratio_val_test)

    # 4 class
    X_val, X_test = X_temp[:split_index_val], X_temp[split_index_val:]
    y_val, y_test = y_temp[:split_index_val], y_temp[split_index_val:]

    transform_list = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset_RNN = CarRacingDataset_RNN(X_train, y_train, sequence_length=5, transform=transform_list)
    train_loader_RNN = DataLoader(train_dataset_RNN, batch_size=32, shuffle=True)

    val_dataset_RNN = CarRacingDataset_RNN(X_val, y_val, sequence_length=5, transform=transform_list)
    val_loader_RNN = DataLoader(val_dataset_RNN, batch_size=32, shuffle=False)

    test_dataset_RNN = CarRacingDataset_RNN(X_test, y_test, sequence_length=5, transform=transform_list)
    test_loader_RNN = DataLoader(test_dataset_RNN, batch_size=32, shuffle=False)

    val_losses, train_losses, train_accs, val_accs = [], [], [], []

    if TRAIN_4:

        best_acc = 0
        for i in range(EPOCH):
            train_loss, train_acc = train(model, train_loader_RNN, loss_fn, optimizer)
            val_loss, val_acc = validation(model, val_loader_RNN, loss_fn)

            if val_acc > best_acc:  # best accuracy out of all epochs
                best_acc = val_acc
                torch.save(model.state_dict(), 'weights/modelLSTM_new.pth')

                # Load the best model weights
            print("Saving best model with accuracy: ", best_acc)

            train_losses.append(train_loss)
            train_accs.append(train_acc)

            val_losses.append(val_loss)
            val_accs.append(val_acc)

    if PLOT:
        # loss plot
        plt.figure(figsize=(15, 60))

        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.legend()
        plt.title(f'Car Data loss plot, BATCH SIZE={BATCH_SIZE}, EPOCH = {EPOCH}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.show()

        # accuracy plot

        plt.figure(figsize=(10, 40))

        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.title(f'Car Data accuracy plot, BATCH SIZE={BATCH_SIZE}, EPOCH = {EPOCH}')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.show()

    if TEST_4:

        if os.path.exists('weights/modelLSTM_new.pth'):
            model.load_state_dict(torch.load('weights/modelLSTM_new.pth'))
        acc, loss = test(model, test_loader_RNN, loss_fn)
        print("Test accuracy(4 class): ", acc)


if __name__ == "__main__":
    main()
