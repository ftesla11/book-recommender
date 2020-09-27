import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def load_data(books_path, ratings_path):
    books = pd.read_csv(books_path, sep=';', error_bad_lines=False, encoding='latin-1',  warn_bad_lines=False, low_memory=False)
    ratings = pd.read_csv(ratings_path, sep=';', error_bad_lines=False, encoding='latin-1',  warn_bad_lines=False, low_memory=False)
    
    # Drop irrelevant features
    books = books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1)

    # Fix incorrect cells
    books.loc[books['ISBN'] == '9627982032', 'Book-Author'] = 'Other'
    books.loc[books['ISBN'] == '078946697X', 'Book-Author'] = 'Michael Teitelbaum'
    books.loc[books['ISBN'] == '078946697X', 'Year-Of-Publication'] = '2000'
    books.loc[books['ISBN'] == '078946697X', 'Publisher'] = 'DK Publishing Inc'

    books.loc[books['ISBN'] == '2070426769', 'Book-Author'] = 'Jean-Marie Gustave'
    books.loc[books['ISBN'] == '2070426769', 'Year-Of-Publication'] = '2003'
    books.loc[books['ISBN'] == '2070426769', 'Publisher'] = 'Gallimard'

    books.loc[books['ISBN'] == '0789466953', 'Book-Author'] = 'James Buckley'
    books.loc[books['ISBN'] == '0789466953', 'Year-Of-Publication'] = '2000'
    books.loc[books['ISBN'] == '0789466953', 'Publisher'] = 'DK Publishing Inc'

    # Check for missing values in any of the columns
    ratings.loc[ratings['User-ID'].isnull() | ratings['ISBN'].isnull() | ratings['Book-Rating'].isnull()]

    # Eliminate implicit feedback
    ratings = ratings[ratings['Book-Rating'] != 0]

    # Keep only ratings that correspond to user and book datasets
    ratings = ratings[ratings['ISBN'].isin(books['ISBN'])]

    # Dataset too large for pivoting
    # Reduce the dataset, filter out books with less than 5 ratings
    popular_books = ratings['ISBN'].value_counts()
    popular_books = popular_books[popular_books > 5]
    ratings = ratings[ratings['ISBN'].isin(popular_books.index)]
    
    return books, ratings

def train_model(ratings_matrix):
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(ratings_matrix)
    return model


def get_ratings_matrix(ratings):
    # Create a pivoted dataframe with user id, isbn and corresponding ratings
    ratings_matrix = ratings.pivot(index='ISBN', columns='User-ID', values='Book-Rating')
    # Replace NaN with 0s for the machine learning model
    ratings_matrix.fillna(value=0, inplace=True)
    return ratings_matrix

# Returns the book title, author, and the cosine similarity of the book compared to the givn book
def recommended_books(model, book, books, ratings_matrix, n_recommendations):
    target_book = books.loc[books['ISBN'] == book, 'Book-Title'].values[0]
    target_author = books.loc[books['ISBN'] == book, 'Book-Author'].values[0]
    target_publisher = books.loc[books['ISBN'] == book, 'Publisher'].values[0]
    target_year = books.loc[books['ISBN'] == book, 'Year-Of-Publication'].values[0]
    recommended_books = []

    # n_neighbors = number of recommended books
    try:   
        distances, indices = model.kneighbors(ratings_matrix.loc[book].values.reshape(1,-1), n_neighbors = n_recommendations+1)
    except:
        print('I do not have this many recommendations for this book.\n')
        return False
        
    cosine_similarity = 1-distances.flatten()
    
    print("Given Book: {} - Author: {} - Published by {} {}\n".format(target_book, target_author, target_publisher, target_year))
    print("Recommended Books: ")
    
    for i in range(1, len(indices.flatten())):
        isbn = ratings_matrix.iloc[indices.flatten()[i]].name
        book_title = books.loc[books['ISBN'] == isbn, 'Book-Title'].values[0]
        book_author = books.loc[books['ISBN'] == isbn, 'Book-Author'].values[0]
        book_year = books.loc[books['ISBN'] == isbn, 'Year-Of-Publication'].values[0]
        book_publisher = books.loc[books['ISBN'] == isbn, 'Publisher'].values[0]
        recommended_books.append([book_title, book_author, book_publisher, book_year])
        print("{}) {} - Author: {} - Published by {} {}\n".format(i, book_title, book_author, book_publisher, book_year))
    return recommended_books

# Return a matrix consisting of books with the given name
def find_book(name, books, ratings_matrix):
    book_list = books.loc[books['Book-Title'].str.contains(name, case=False)]
    valid_books = book_list[book_list['ISBN'].isin(ratings_matrix.index)]
    return valid_books


