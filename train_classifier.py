import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



def separate_data(data, labels):

    one_hand_classes = ['c', 'i', 'l', 'o', 'u', 'v']
    two_hand_classes = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'w', 'x', 'y', 'z']
    one_hand_data = []
    one_hand_labels = []
    two_hand_data = []
    two_hand_labels = []
    
    for d, l in zip(data, labels):
        if l in one_hand_classes:
            one_hand_data.append(d)
            one_hand_labels.append(l)
        elif l in two_hand_classes:
            two_hand_data.append(d)
            two_hand_labels.append(l)

    return np.array(one_hand_data), np.array(one_hand_labels), np.array(two_hand_data), np.array(two_hand_labels)



def train_classifier(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, accuracy



def save_classifier(clf, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(clf, f)



def main():

    #Loading and arranging the data
    data_dict = pickle.load(open('./data.pickle', 'rb'))
    data = data_dict['data']
    labels = data_dict['labels']
    
    #Separating the one hand data and the two hand data
    one_hand_data, one_hand_labels, two_hand_data, two_hand_labels = separate_data(data, labels)
    
    #Training one-hand classifier
    one_hand_clf, one_hand_accuracy = train_classifier(one_hand_data, one_hand_labels)
    print(f"One-hand classifier accuracy: {one_hand_accuracy:.2f}")
    
    #Training two-hand classifier
    two_hand_clf, two_hand_accuracy = train_classifier(two_hand_data, two_hand_labels)
    print(f"Two-hand classifier accuracy: {two_hand_accuracy:.2f}")
    
    #Saving classifiers
    save_classifier(one_hand_clf, 'one_hand_classifier.pkl')
    save_classifier(two_hand_clf, 'two_hand_classifier.pkl')
    
    print("Classifiers trained and saved successfully.")


if __name__ == "__main__":
    main()