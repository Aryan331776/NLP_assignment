import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#Question 1

'''
# part 1: Generate the array
np.random.seed(0)  
arr1 = np.random.randint(1, 51, size=(5, 4))  
print("Original Array:")
print(arr1)

# part 2: Extract anti-diagonal (from top-right to bottom-left)
anti_diagonal = [int(arr1[i][3-i]) for i in range(0,4)]
print("\nAnti-diagonal elements:")
print(anti_diagonal)

# part 3: Maximum value in each row
row_maxima = np.max(arr1, axis=1)
print("\nMaximum value in each row:")
print(row_maxima)

# part 4: Elements <= overall mean
overall_mean = np.mean(arr1)
filtered_elements = arr1[arr1 <= overall_mean]
print("\nElements less than or equal to overall mean:")
print(filtered_elements)

# part 5: Function to perform boundary traversal
def numpy_boundary_traversal(matrix):
    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        return []
    
    boundary = []
    for j in range(cols):
        boundary.append(matrix[0, j])
    for i in range(1, rows):
        boundary.append(matrix[i, cols - 1])
    if rows > 1:
        for j in range(cols - 2, -1, -1):
            boundary.append(matrix[rows - 1, j])
    if cols > 1:
        for i in range(rows - 2, 0, -1):
            boundary.append(matrix[i, 0])
    return boundary
    




#Question 2

arr2 = np.random.uniform(0, 10, 20)

# part 2: Print and round all elements to two decimal places
rounded_arr = np.round(arr2, 2)
print("Rounded Array:")
print(rounded_arr)

# part 3: Calculate and print min, max, and median
print("\nStatistics:")
print("Minimum:", np.min(rounded_arr))
print("Maximum:", np.max(rounded_arr))
print("Median:", np.median(rounded_arr))

# part 4: Replace elements less than 5 with their squares
modified_arr = np.where(rounded_arr < 5, np.round(rounded_arr ** 2, 2), rounded_arr)
print("\nModified Array (elements < 5 squared):")
print(modified_arr)

# part 5: Function to sort in alternating pattern
def numpy_alternate_sort(array):
    sorted_array = np.sort(array)
    result = []
    i, j = 0, len(sorted_array) - 1
    while i <= j:
        if i == j:
            result.append(sorted_array[i])
        else:
            result.append(sorted_array[i])
            result.append(sorted_array[j])
        i += 1
        j -= 1
    return np.array(result)

    

#Question - 3


names = ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Hannah', 'Ivan', 'Julia']
subject = 'Math'
subjects = [subject] * 10  # All subjects are 'Math'

np.random.seed(0)  # For reproducibility
scores = np.random.randint(50, 101, size=10)

df = pd.DataFrame({
    'Name': names,
    'Subject': subjects,
    'Score': scores,
    'Grade': ''  
})

# ------------------------------
# part-2: Assign grades based on scores:
#         A (90–100), B (80–89), C (70–79), D (60–69), F (below 60)
# ------------------------------

def assign_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

df['Grade'] = df['Score'].apply(assign_grade)

# ------------------------------
# part-3: Print the DataFrame sorted by Score in descending order
# ------------------------------

print("Sorted by Score (descending):")
print(df.sort_values(by='Score', ascending=False))

# ------------------------------
# part-4: Calculate and print the average score for the subject
# ------------------------------

print("\nAverage Score for Math:")
print(df['Score'].mean())

# ------------------------------
# part-5: Write a Python function pandas_filter_pass(dataframe)
#         that returns a new DataFrame with grades A or B only
# ------------------------------

def pandas_filter_pass(dataframe):
    return dataframe[dataframe['Grade'].isin(['A', 'B'])]

filtered_df = pandas_filter_pass(df)

print("\nStudents with Grade A or B:")
print(filtered_df)


# Question - 4

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------
# Step 1: Create synthetic dataset (100 reviews: 50 positive, 50 negative)
# ------------------------------
positive_reviews = [
    "Amazing movie, I loved it!", "Great acting and wonderful plot.",
    "This film was fantastic!", "An outstanding performance!",
    "Really touching story.", "Beautifully made and emotional.",
    "Highly recommend this movie!", "A true masterpiece.",
    "Loved every minute of it.", "The cast did a brilliant job.",
    "Spectacular visuals and direction.", "Truly unforgettable experience.",
    "Incredible storytelling.", "Heartwarming and inspiring.",
    "A feel-good movie.", "It made me laugh and cry.",
    "The best movie I've seen this year.", "A cinematic gem.",
    "Exceptional in every way.", "Simply perfect.",
    "A must-watch film!", "Impressive from start to finish.",
    "10/10 would watch again.", "Pure brilliance.",
    "A thrilling and emotional ride.", "Completely blew me away.",
    "An instant classic.", "Flawless execution.",
    "It exceeded my expectations.", "Loved the chemistry between the actors.",
    "Perfect pacing and storytelling.", "The soundtrack was amazing.",
    "A great film for the whole family.", "A joy to watch.",
    "Powerful and moving.", "Entertaining and thought-provoking.",
    "The director did an amazing job.", "A very satisfying movie.",
    "Funny, smart, and well-made.", "It's a breath of fresh air.",
    "Strong performances by the whole cast.", "A beautiful and compelling film.",
    "Full of heart and charm.", "I can't wait to watch it again.",
    "It left me speechless.", "The emotions felt real and genuine.",
    "Superb film.", "Great from beginning to end.",
    "A wonderfully told story.", "Everything about it was great.",
]

negative_reviews = [
    "Terrible movie, I hated it.", "Worst film I've seen.",
    "Awful acting and terrible plot.", "Completely boring and predictable.",
    "A waste of time.", "Couldn't even finish it.",
    "Disappointing and dull.", "Nothing good about this movie.",
    "Bad script and worse direction.", "I regret watching this.",
    "Painfully slow and uninteresting.", "This film was a mess.",
    "No emotional connection at all.", "Flat characters and poor acting.",
    "Ridiculous storyline.", "Tried too hard and failed.",
    "Unbearable to watch.", "I don't recommend it.",
    "Weak plot and terrible pacing.", "Cringe-worthy dialogue.",
    "It made no sense.", "Just bad in every way.",
    "I fell asleep during the movie.", "Plot holes everywhere.",
    "Bad special effects.", "I walked out of the theater.",
    "It lacked everything a good movie needs.", "It was just noise.",
    "Terribly executed.", "It was painful to sit through.",
    "This movie was pure garbage.", "An insult to cinema.",
    "It failed on all fronts.", "A predictable disappointment.",
    "Cheap and lazy production.", "The acting was robotic.",
    "So bad its almost funny.", "How did this get made?",
    "Dont waste your money.", "Forgettable and bland.",
    "I didnt care about any of the characters.", "It was nonsense.",
    "Fails to entertain.", "Low quality in every aspect.",
    "No depth at all.", "Too many clichés.",
    "Extremely overrated.", "Unwatchable garbage.",
    "It was a joke.", "A cinematic disaster.",
]
print(len(negative_reviews))
print(len(positive_reviews))

# Combine into a DataFrame
reviews = positive_reviews + negative_reviews
sentiments = ['positive'] * 50 + ['negative'] * 50

df = pd.DataFrame({
    'Review': reviews,
    'Sentiment': sentiments
})

# ------------------------------
# Step 2: Tokenize the reviews using CountVectorizer
# ------------------------------
vectorizer = CountVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(df['Review'])

# ------------------------------
# Step 3: Split into training and test sets
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, df['Sentiment'], test_size=0.2, random_state=42)

# ------------------------------
# Step 4: Train a Multinomial Naive Bayes classifier
# ------------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# ------------------------------
# Step 5: Evaluate with accuracy, precision, recall, F1
# ------------------------------
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# Detailed classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

# ------------------------------
# Step 6: Function to predict sentiment for new reviews
# ------------------------------
def predict_review_sentiment(model, vectorizer, review):
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)[0]
    return prediction

# Example usage
example_review = "The movie was emotional and brilliantly directed."
print("\nExample Review Prediction:")
print(f"Review: {example_review}")
print(f"Predicted Sentiment: {predict_review_sentiment(model, vectorizer, example_review)}")
'''

# Question- 5

# ----------------------------
# 1. Create a synthetic dataset
# ----------------------------

good_feedback = [
    "Excellent product, works perfectly!", "Very satisfied with this purchase.",
    "High quality and reliable.", "Fast shipping and great value.",
    "Exceeded my expectations.", "Would definitely recommend.",
    "Product matches the description.", "Impressive performance.",
    "Affordable and effective.", "Really pleased with the outcome.",
    "Superb build and usability.", "Five stars!", "Love this product!",
    "Worth every penny.", "Does exactly what it promises.",
    "Customer service was helpful.", "Packaging was neat and secure.",
    "Delivered on time and intact.", "Smooth experience overall.",
    "Will buy again.", "Totally satisfied.", "A dependable product.",
    "Great support from the seller.", "Very intuitive to use.",
    "Perfect for my needs.", "Feels premium.", "Top-notch quality.",
    "Easy to install and operate.", "Definitely worth it.",
    "This made my life easier.", "I’d give more stars if I could.",
    "Performs even better than expected.", "The design is sleek.",
    "Very durable and long-lasting.", "Reliable and consistent.",
    "Battery life is amazing.", "Simple and efficient.",
    "Highly functional product.", "No issues at all.",
    "Love the features!", "Very happy with this.",
    "Great results every time.", "Product feels solid.",
    "Easy to clean and maintain.", "Good value for money.",
    "Came with everything I needed.", "Very responsive device.",
    "Great investment.", "Exactly what I was looking for.",
    "Totally recommend it.", "Loved it!"
]

bad_feedback = [
    "Terrible product, don’t buy it.", "Completely useless.",
    "Very disappointed.", "Did not work as advertised.",
    "Stopped working after a week.", "Waste of money.",
    "Feels cheap and fragile.", "Customer support is unresponsive.",
    "Arrived broken.", "Would not recommend.",
    "Regret buying this.", "Low quality material.",
    "Performance is terrible.", "Does not do what it says.",
    "Overpriced and underdelivers.", "Battery drains quickly.",
    "Hard to use and confusing.", "Not worth the price.",
    "Looks nothing like the photos.", "Shipping was super late.",
    "Doesn’t last long.", "Totally unreliable.",
    "Frequent crashes and bugs.", "Bad build quality.",
    "Instructions were unclear.", "Very poor packaging.",
    "Missing parts in the box.", "Too loud and annoying.",
    "Doesn't fit as expected.", "Returned it immediately.",
    "Feels very flimsy.", "No noticeable benefits.",
    "Very cheap plastic.", "Just terrible.",
    "Extremely disappointed.", "It broke instantly.",
    "Seller never replied.", "Useless manual.",
    "Design is flawed.", "Build feels poor.",
    "Setup was a nightmare.", "Too complicated to use.",
    "Performance was a letdown.", "Not durable at all.",
    "Awful experience overall.", "Didn't meet expectations.",
    "Lacking basic features.", "Uncomfortable to use.",
    "Unstable and glitchy.", "Worst purchase ever.",
    "Do not waste your money."
]

texts = good_feedback + bad_feedback
labels = ['good'] * 50 + ['bad'] * 50

# Shuffle dataset
combined = list(zip(texts, labels))
random.shuffle(combined)
texts, labels = zip(*combined)

# ----------------------------
# 2. Preprocess using TfidfVectorizer
# ----------------------------

vectorizer = TfidfVectorizer(max_features= 1200 , stop_words='english', lowercase=True)
X = vectorizer.fit_transform(texts)
y = labels

# ----------------------------
# 3. Split dataset
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ----------------------------
# 4. Train Logistic Regression
# ----------------------------

model = LogisticRegression(class_weight = 'balanced')
model.fit(X_train, y_train)

# ----------------------------
# 5. Evaluate performance
# ----------------------------

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['bad', 'good']))

# ----------------------------
# 6. Vectorization Function
# ----------------------------

def text_preprocess_vectorize(texts, vectorizer):
    
    return vectorizer.transform(texts)


example = ["Product was easy to use and really helpful."]
vec_example = text_preprocess_vectorize(example, vectorizer)
print("\nPrediction for example:", model.predict(vec_example)[0])













