import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('E:/Internship/internship hemachandra/Spam_Sms_Detection/Dataset/spam.csv', encoding='latin-1')

# Select only the relevant columns and rename them
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Check for NaN values and drop them
data.dropna(inplace=True)

# Check if any rows were dropped
if data.empty:
    print("The dataset is empty after cleaning. Please check your data.")
else:
    # Convert labels to binary (ham: 0, spam: 1)
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Create and train the Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = nb_model.predict(X_test_tfidf)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Print classification report
    print(classification_report(y_test, y_pred))

    # Prepare output data with only spam messages
    output_data = pd.DataFrame({
        'label': ['spam' if label == 1 else 'ham' for label in y_pred],
        'message': X_test,
        'empty1': '',  # Adding empty columns
        'empty2': '',
        'empty3': ''
    })

    # Filter to keep only spam messages
    output_data = output_data[output_data['label'] == 'spam']

# Save the output to a CSV file with ',' delimiter in the specified path
    output_file_path = 'E:/Internship/internship hemachandra/Spam_Sms_Detection/spam_detection_output.csv'
    output_data.to_csv(output_file_path, index=False, sep=',')
    print(f"\nPredictions saved to {output_file_path}")
