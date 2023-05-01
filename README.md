Book Recommender System
This application is a book recommender system that recommends similar books to the user's selected book. The app is built using Streamlit, a Python library for building web apps for machine learning and data science. The app uses a k-nearest neighbors algorithm to find the top 5 most similar books to the user's selection.

Installation
To use this app, you need to have Python 3.8 or higher installed. You also need to have the following Python libraries installed:

Streamlit
Pandas
Scikit-learn

After installing the required libraries, you need to download the dataset used by the app. The dataset includes three CSV files: Books.csv, Ratings.csv, and Users.csv.
After downloading the dataset, place the three CSV files in the same directory as the app.py file.

Once the app is running, you can select a book from the dropdown menu to see similar books. 
The app will display the top 5 most similar books to the user's selection, along with their title, author, year of publication, publisher, and an image of the book cover.

App Workflow
The app uses the following workflow to recommend similar books to the user's selection:

Load the dataset, which includes information about books, ratings, and users.
Preprocess the data by filtering out users and books with too few ratings, creating a pivot table, and filling in missing values with zeros.
Fit a k-nearest neighbors algorithm to the pivot table, using cosine similarity as the distance metric.
Define a function to get similar books, which takes a book name as input and returns the top 5 most similar books.
Define a function to get book information, which takes a book name as input and returns information about the book from the dataset.
Define the Streamlit app, which creates a dropdown menu of books and displays the top 5 most similar books to the user's selection.
