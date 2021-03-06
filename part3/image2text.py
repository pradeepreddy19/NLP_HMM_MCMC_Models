#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
#
# Authors:  harmohan, jzayatz , prokkam
# (based on skeleton code by D. Crandall, Oct 2020)
#
import math
from PIL import Image
import sys

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25
TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [["".join(['*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg + CHARACTER_WIDTH)]) for y in
                    range(0, CHARACTER_HEIGHT)], ]
    return result


def load_training_letters(fname):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS))}


if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)


#simple naive bayes classifier
def simple_method(test_letters):
    predicted_string = ''
    n=CHARACTER_WIDTH * CHARACTER_HEIGHT #total pixels
    for alphabet in test_letters:
        char_probabilities = {}
        for character in TRAIN_LETTERS:
            match_count, no_match_count, blankspace_match_count = 0, 0, 0
            for i in range(len(alphabet)):
                for j in range(len(alphabet[i])):
                    if alphabet[i][j] == train_letters[character][i][j] == '*': #matching stars
                        match_count += 1
                    elif alphabet[i][j] == train_letters[character][i][j] == ' ': #matching spaces
                        blankspace_match_count += 1
                    else:
                        no_match_count += 1 #pixels that do not match
            #weighted counts fine tuned
            char_probabilities[character] = (0.95 * match_count + 0.04 * blankspace_match_count + 0.01 * no_match_count) / n 
        #concatenating string based on max computed prob (likelihood)      
        predicted_string +=  max(char_probabilities, key=char_probabilities.get) # Ref- https://www.kite.com/python/answers/how-to-find-the-max-value-in-a-dictionary-in-python
        
    return predicted_string

print("Simple: " + simple_method(test_letters)) #prints simple output

#function to compute initial char counts from train.txt
def calc_initial_char_count(final_sentence_data):
    initial_state_prob_dict={}
    for sentence_data in final_sentence_data:
        for each in sentence_data[0]:
            if each[0] in train_letters: #making sure it is valid char 
                if each[0] in initial_state_prob_dict:
                    initial_state_prob_dict[each[0]] += 1
                else:
                    initial_state_prob_dict[each[0]] = 1
    return initial_state_prob_dict

#function to compute transition counts
def calc_transition_count(final_sentence_data):
    transition_prob_dict={}
    for sentence_data in final_sentence_data:
        data = " ".join(sentence_data)
        for i in range(1,len(data)):  
            first = i-1
            next = i         
            if (data[first], data[next]) in transition_prob_dict:
                transition_prob_dict[data[first], data[next]] += 1
            else:
                transition_prob_dict[data[first], data[next]] = 1
    return transition_prob_dict

#function to compute count of char for denominator 
def calc_char_count(sentences_list):
    char_count_dict = {}
    for k in range(len(sentences_list)):
        for l in range(len(sentences_list[k])):
            if sentences_list[k][l] in char_count_dict:
                char_count_dict[sentences_list[k][l]] += 1
            else:
                char_count_dict[sentences_list[k][l]] = 1
    return char_count_dict

#same simple method to calculate emission probs as mentioned above with fine tuned weights
def calc_emission_prob(aplhabet, test_letters):
    n=CHARACTER_WIDTH * CHARACTER_HEIGHT
    match_count, no_match_count, blankspace_match_count = 0, 0, 0
    for h in range(CHARACTER_HEIGHT):
        for w in range(CHARACTER_WIDTH):
            if test_letters[h][w] == train_letters[aplhabet][h][w] == '*':
                match_count += 1
            elif test_letters[h][w] ==  train_letters[aplhabet][h][w] == ' ':
                blankspace_match_count += 1
            elif test_letters[h][w] != train_letters[aplhabet][h][w]  :
                no_match_count += 1
    emmision = (0.95 * match_count + 0.04 * blankspace_match_count + 0.01 * no_match_count) / n
    return math.log(emmision)


#training over train.txt from part1 to compute initial, transition and emission probs
def train():
    final_sentence_data = []
    for sentence in open(train_txt_fname, 'r'):
        sentence_carrier=sentence.split()
        #taking alternate words in order to avoid meaningless POS tags in the dataset
        sentence_data = tuple([sentence_carrier[i] for i in range(len(sentence_carrier)) if i%2==0])
        final_sentence_data += [sentence_data]


    sentences_list=[]
    for each in final_sentence_data:
        sentences_list.append(" ".join(each))

   
    initial_state_prob_dict=calc_initial_char_count(final_sentence_data)


    transition_prob_dict=calc_transition_count(final_sentence_data)

    char_count_dict=calc_char_count(sentences_list)    

    #calculate trans prob
    for first in TRAIN_LETTERS:
        for next in TRAIN_LETTERS:
            if (first, next) in transition_prob_dict:
                transition_prob_dict[first, next] = transition_prob_dict[first, next] / char_count_dict[first]
               
            else:
                transition_prob_dict[first, next] = 0.0000000001 #allocating 1^-10 small value for transitions not observed
               

    return [initial_state_prob_dict, char_count_dict, transition_prob_dict]


def HMM_Viterbi(test_letters, transition_prob_dict):
    V_table = []
    V_0 = []
    for i in range(0,len(TRAIN_LETTERS)):   #viterbi first iteration value     
        V_0.append(calc_emission_prob(TRAIN_LETTERS[i], test_letters[0]))
    V_table.append(V_0)

    for i in range(1, len(test_letters)): #viterbi second iteration onwards
        V_i = []
        for j in range(len(TRAIN_LETTERS)):
            em_prob = (calc_emission_prob(TRAIN_LETTERS[j], test_letters[i])) 
            #variable to store max trans prob
            trans_max_value = 0 
            for k in range(0,len(TRAIN_LETTERS)):
                trans_prob=0
                if (TRAIN_LETTERS[j], TRAIN_LETTERS[k]) in transition_prob_dict.keys():
                    trans_prob = math.log(transition_prob_dict[(TRAIN_LETTERS[j], TRAIN_LETTERS[k])])
                else:
                    trans_prob = math.log(0.0000000001) #unobserved transitions 
                #the max of all states will be stored at the end of the inner for loop
                trans_max_value=max(V_table[i - 1][k] + trans_prob,trans_max_value)
            V_i_value = trans_max_value + em_prob
            V_i.append(V_i_value)
        V_table.append(V_i)
    return V_table
    


#training
[initial_state_prob_dict, char_count_dict, transition_prob_dict] = train()
viterbi=HMM_Viterbi(test_letters, transition_prob_dict)
#putting together string from viterbi table
predicted_string = ''
for a in range(0,len(test_letters)):        
    small = -999999
    pos = 0
    for b in range(0,len(TRAIN_LETTERS)):
        if small < viterbi[a][b]:
            small = viterbi[a][b]
            pos = b
    predicted_string += TRAIN_LETTERS[pos]

print("   HMM: " + predicted_string)
