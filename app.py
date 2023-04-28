import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors



# Load data
df_books = pd.read_csv("books.csv", low_memory=False)
df_ratings = pd.read_csv("ratings.csv")
df_users = pd.read_csv("users.csv")

merged_df = pd.merge(df_books, df_ratings, on='ISBN')
new_df = merged_df[['Book-Title', 'User-ID', 'Book-Rating']]
data_1 = new_df.groupby('User-ID').count()['Book-Rating'] > 200
imp_users = data_1[data_1].index
ratings_filtered = new_df[new_df['User-ID'].isin(imp_users)]
data_2 = ratings_filtered.groupby('Book-Title').count()['Book-Rating'] >= 50
imp_books = data_2[data_2].index
final_ratings = ratings_filtered[ratings_filtered['Book-Title'].isin(imp_books)]
pivot_table = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pivot_table.fillna(0, inplace=True)
model = NearestNeighbors(algorithm='brute', metric='cosine')
model.fit(pivot_table)

# Define function to get similar books
def get_similar_books(book_name):
    # Check the shape of the input pivot table
    if pivot_table.shape[1] < model.n_features_in_:
        # Add missing columns and fill with zeros
        missing_cols = model.n_features_in_ - pivot_table.shape[1]
        for i in range(missing_cols):
            pivot_table[pivot_table.columns[-1]+1] = 0

    # Find the index of the book
    book_index = pivot_table.index.get_loc(book_name)

    # Find the indices of the top 5 most similar books using the clustering model
    distances, indices = model.kneighbors(pivot_table.iloc[book_index:book_index+1], n_neighbors=6)
    similar_books_indices = indices.flatten()[1:]

    # Get the names of the top 5 similar books
    similar_books = pivot_table.iloc[similar_books_indices].index.tolist()

    return similar_books

# Define function to get book information
def get_book_info(book_name):
    book_info = df_books.loc[df_books['Book-Title'] == book_name]
    return book_info

# Define Streamlit app
def app():
    st.title("Book Recommender system")
    st.write("Select a book from the dropdown menu to see similar books")

    # Create dropdown menu of books
    book_names = pivot_table.index.tolist()
    selected_book = st.selectbox("Select a book", book_names)

    # Get similar books
    similar_books = get_similar_books(selected_book)

    # Display similar books
    st.write("Similar Books:")
    for book in similar_books:
        book_info = get_book_info(book)
        st.write(f"Title: {book_info['Book-Title'].values[0]}")
        st.write(f"Author: {book_info['Book-Author'].values[0]}")
        st.write(f"Year of Publication: {book_info['Year-Of-Publication'].values[0]}")
        st.write(f"Publisher: {book_info['Publisher'].values[0]}")
        st.image(book_info['Image-URL-M'].values[0], width=100)

if __name__ == '__main__':
    app()
