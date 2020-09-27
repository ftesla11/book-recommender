import recommender

def check_response():
    command = input('\nWould you like to get another recommendation? [y/n]\n').lower()
    while True:
        if command == 'n':
            quit()
        elif command == 'y':
            break
        else:
            command = input('I did not understand your command. Please type y or n\n')
        
            
def execute():
    while True:
        user_book = input("\nEnter your favourite book:\n")
        
        valid_books = recommender.find_book(user_book, books, ratings_matrix)

        if valid_books.shape[0] < 1: 
            print('Your book was not found in my dataset of 200+ books.')
            check_response()
            continue
 
        target_book = valid_books.iloc[1,0]
        
        n_recommendations = input("\nI found your book! \nHow many recommendations would you like?\n")

        print('\nHere are your {} recommended books:\n'.format(n_recommendations))
        recommended_books = recommender.recommended_books(model, target_book, books, ratings_matrix, int(n_recommendations))
        check_response()
       
    
print('-----Loading books-----')
books, ratings = recommender.load_data('book-crossing/books.csv', 'book-crossing/ratings.csv')
print('-----Loading tools-----')
ratings_matrix = recommender.get_ratings_matrix(ratings)
print('-----Loading model-----\n')
model = recommender.train_model(ratings_matrix)
execute()