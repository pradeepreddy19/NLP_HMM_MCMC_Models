###################################
# CS B551 Spring 2021, Assignment #3
#
# Your names and user ids:
# harmohan -- Harini Mohansundaram 
# jzayatz  -- Joy Zayatz
# prokkam  -- Pradeep Reddy Rokkam
#
# (Based on skeleton code by D. Crandall)
#


import random
import math


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        # print("The label value is ",label)
        if model == "Simple":
            # print(sentence)
            self.pos_of_sentence=[]
            self.pos_prob_of_sentece=[]
            # print("The type of sentence is ", type(sentence))

            for word in sentence:
                word_prob={'adj':{}, 'adv': {}, 'adp':{}, 'conj':{}, 'det':{}, 'noun':{}, 'num':{}, 'pron':{},'prt':{}, 'verb':{}, 'x':{},'.':{} }
                # print("Word in Sentence is :",word)
                for pos in self.pos_prob.keys():
                    if word in self.pos_word_prob[pos].keys():
                        word_prob[pos]=self.pos_word_prob[pos][word]*self.pos_prob[pos]
                    else:
                         word_prob[pos]=0

                self.pos_prob_of_sentece.append(max(word_prob.values()))
            
            return( sum([math.log(x) for x in self.pos_prob_of_sentece if x!=0]))

            
            # # Throwing an Zero Division Error So that code stops ( Please Remove this before the submission)
            # print(25/0)
            return -999
        elif model == "HMM":
            return -999
        elif model == "Complex":
            return -999
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        print(len(data))
        pos_count={}

        # Creating a dictionary that has parts of speech as keys  ad values are again dictionaries (This dictionary has the words as the keys and the values will be their count of given word for the given parts of speech )
        pos_word_count={}

        # Creating a dictionary to capture the initial state i.e. The POS of the first word in each of the sentence
        hmm_S0_count= {}

        # Creating a dictionary to capture the transition probilities for the state 
        hmm_Si_to_Sj_count= {}

        ## The following loop updates the POS count  and POS_WORD_COUNT
        for i in range(len(data)):

            line= data[i]


            for j in range(len(line[0])):


                ###The following if statement captures the S0 (State 0 - POS of te first word ) probabilities
                if j==0:
                    if line[1][0] in hmm_S0_count.keys():
                        hmm_S0_count[line[1][0]] += 1
                    else:
                        hmm_S0_count[line[1][j]] = 1

                 ###The following if statement captures the transition probabilities

                else:
                    if line[1][j-1] in hmm_Si_to_Sj_count.keys():
                        if line[1][j] in hmm_Si_to_Sj_count[line[1][j-1]].keys():
                            hmm_Si_to_Sj_count[line[1][j-1]][line[1][j]] += 1
                        else:
                            hmm_Si_to_Sj_count[line[1][j-1]][line[1][j]] = 1

                    else:
                        hmm_Si_to_Sj_count[line[1][j-1]]= {}
                        hmm_Si_to_Sj_count[line[1][j-1]][line[1][j]] = 1                    

                #############################################

                ## Updating Parts of Speech Count #############
                if line[1][j] in pos_count.keys():
                    pos_count[line[1][j]] += 1
                else:
                    pos_count[line[1][j]] = 1
                
                ## Updating Parts of Speech Word Count ##############

                if line[1][j] in pos_word_count.keys():
                    if line[0][j] in  pos_word_count[line[1][j]].keys():
                        pos_word_count[line[1][j]][line[0][j]]+=1
                    else:
                        pos_word_count[line[1][j]][line[0][j]]=1
                else:
                    pos_word_count[line[1][j]]={}
                    pos_word_count[line[1][j]][line[0][j]]=1
        
        #### Get the initial state probabilties. This will be used in HMM and Viterbi 
        hmm_S0_prob = {}

        total = sum(hmm_S0_count.values())

        for each in hmm_S0_count.keys():
            hmm_S0_prob[each]=hmm_S0_count[each]/total

        #### Get the transition probabailities for the states. This will be used in HMM and Viterbi 
        hmm_Si_to_Sj_prob = {'adj':{}, 'adv': {}, 'adp':{}, 'conj':{}, 'det':{}, 'noun':{}, 'num':{}, 'pron':{},'prt':{}, 'verb':{}, 'x':{},'.':{} }

        for each in hmm_Si_to_Sj_count.keys():

            total = sum(hmm_Si_to_Sj_count[each].values())
    

            for pos in hmm_Si_to_Sj_count[each].keys():
                hmm_Si_to_Sj_prob[each][pos]=hmm_Si_to_Sj_count[each][pos]/total

        #Get the prior proabailites for the Parts of Speech:
        self.pos_prob={}

            ## Get the overall pos count 
        count=0
        
        for pos in pos_count.keys():
            count= count+pos_count[pos]
            
            ## Using the count obatined in the previous step, divide each value of pos_count dictionary to get the probability
        for pos in pos_count.keys():
            self.pos_prob[pos]=pos_count[pos]/count
       
        #Get the prior proabailites (Likelihood Values) for the Parts of Speech:
        self.pos_word_prob={'adj':{}, 'adv': {}, 'adp':{}, 'conj':{}, 'det':{}, 'noun':{}, 'num':{}, 'pron':{},'prt':{}, 'verb':{}, 'x':{},'.':{} }
       
            ## Using the count from pos_count dictionary we will get the likelihood values
        for pos in pos_word_count.keys():
            count= pos_count[pos]
           
            for word in pos_word_count[pos].keys():
                self.pos_word_prob[pos][word]=pos_word_count[pos][word]/count

        print(hmm_S0_prob)
        # print(hmm_Si_to_Sj_prob)
        for each in hmm_Si_to_Sj_count.keys():
            print( each," : ", hmm_Si_to_Sj_count[each])
        print("__"*50)
        for each in hmm_Si_to_Sj_prob.keys():
            print( each," : ", hmm_Si_to_Sj_prob[each])

        print(25/0)
        # print(pos_count)
        # print(self.pos_prob)
        # print("__"*50)

        # Validation 
        # Run the following code if you wan to check is the count of wrods in pos_count is equal to pos_word_count
        # for each in pos_word_count.keys():
        #     word_count=0
        #     for word in pos_word_count[each].keys():
        #         word_count=word_count+pos_word_count[each][word]
        #     print(each, ":", word_count)

        
            

    

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        

        # print(sentence)
        self.pos_of_sentence=[]
        # print("The type of sentence is ", type(sentence))

        for word in sentence:
            word_prob={'ADJ':0, 'ADV': 0, 'ADP':0, 'CONJ':0, 'DET':0, 'NOUN':0, 'NUM':0, 'PRON':0,'PRT':0, 'VERB':0, 'X':0,'.':0 }
            # print("Word in Sentence is :",word)
            for pos in self.pos_prob.keys():
                if word in self.pos_word_prob[pos].keys():
                    word_prob[pos]=self.pos_word_prob[pos][word]*self.pos_prob[pos]
                else:
                        word_prob[pos]=0
            # print("The word is ", word)
            # print(word_prob)
            max_prob=0
            for each in word_prob.keys():
                if word_prob[each]>=max_prob:
                    max_prob=word_prob[each]
                    pos_of_word= each
            # print(pos_of_word)
            self.pos_of_sentence.append(pos_of_word)

        return self.pos_of_sentence
        
    def hmm_viterbi(self, sentence):

        return [ "noun" ] * len(sentence)

    def complex_mcmc(self, sentence):
        return [ "noun" ] * len(sentence)



    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

