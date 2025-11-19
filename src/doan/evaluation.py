import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

def predictor(model, test_gen, test_steps, classes):
    y_pred = []
    y_true = test_gen.labels
    class_count = len(classes)
    errors = 0
    preds = model.predict(test_gen, steps=test_steps, verbose=1)
    tests = len(preds)
    for i, p in enumerate(preds):
        pred_index = np.argmax(p)
        true_index = test_gen.labels[i]
        if pred_index != true_index:
            errors += 1
        y_pred.append(pred_index)
    acc = (1 - errors/tests) * 100
    print(f'there were {errors} in {tests} tests for an accuracy of {acc:6.2f}')
    ypred = np.array(y_pred)
    ytrue = np.array(y_true)
    if class_count <= 30:
        cm = confusion_matrix(ytrue, ypred)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(np.arange(class_count)+.5, classes, rotation=90)
        plt.yticks(np.arange(class_count)+.5, classes, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    clr = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print("Classification Report:\n----------------------\n", clr)
    return errors, tests

def save_model(model, subject, accuracy, working_dir):
    acc_str = str(accuracy)
    index = acc_str.rfind('.')
    acc = acc_str[:index + 3]
    save_id = subject + '_' + str(acc) + '.h5'
    model_save_loc = os.path.join(working_dir, save_id)
    model.save(model_save_loc)
    print('model was saved as ', model_save_loc)