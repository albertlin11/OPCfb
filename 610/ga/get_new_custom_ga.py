import numpy as np
from write_model_ga import write_file



filename = "ga_result.txt"

# Read in the file
with open(filename, 'r') as file:
    best_list = []
    # read number only
    for line in file:
        # Split the line into words
        words = line.split()
        
        # Now 'words' is a list of individual words
        # You can iterate over the words and do whatever you need
        for word in words:
            
            word = word.replace('[', '')
            word = word.replace(']', '')
            word = word.replace('.', '')
            if word.isdigit(): # check if the word is a number
                word = int(word)           
                best_list.append(word)

    print(best_list)  

write_file(best_list)

        

