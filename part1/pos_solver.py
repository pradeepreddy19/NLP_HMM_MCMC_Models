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
        self.pos_count={}


        # Creating a dictionary that has parts of speech as keys and values are again dictionaries (This dictionary has the words as the keys and the values will be their count of given word for the given parts of speech )
        self.pos_word_count={}

        # Creating a dictionary to capture the initial state i.e. The POS of the first word in each of the sentence
        self.hmm_S0_count= {}

        # Creating a dictionary to capture the transition probilities for the state 
        self.hmm_Si_to_Sj_count= {}

        ## The following loop updates the POS count  and self.pos_word_count
        for i in range(len(data)):

            line= data[i]


            for j in range(len(line[0])):


                ###The following if statement captures the S0 (State 0 - POS of te first word ) probabilities
                if j==0:
                    if line[1][0] in self.hmm_S0_count.keys():
                        self.hmm_S0_count[line[1][0]] += 1
                    else:
                        self.hmm_S0_count[line[1][j]] = 1

                 ###The following if statement captures the transition probabilities

                else:
                    if line[1][j-1] in self.hmm_Si_to_Sj_count.keys():
                        if line[1][j] in self.hmm_Si_to_Sj_count[line[1][j-1]].keys():
                            self.hmm_Si_to_Sj_count[line[1][j-1]][line[1][j]] += 1
                        else:
                            self.hmm_Si_to_Sj_count[line[1][j-1]][line[1][j]] = 1

                    else:
                        self.hmm_Si_to_Sj_count[line[1][j-1]]= {}
                        self.hmm_Si_to_Sj_count[line[1][j-1]][line[1][j]] = 1                    

                #############################################

                ## Updating Parts of Speech Count (P(S)) #############
                if line[1][j] in self.pos_count.keys():
                    self.pos_count[line[1][j]] += 1
                else:
                    self.pos_count[line[1][j]] = 1
                
                ## Updating Parts of Speech Word Count ( P(W/S)) ##############

                if line[1][j] in self.pos_word_count.keys():
                    if line[0][j] in  self.pos_word_count[line[1][j]].keys():
                        self.pos_word_count[line[1][j]][line[0][j]]+=1
                    else:
                        self.pos_word_count[line[1][j]][line[0][j]]=1
                else:
                    self.pos_word_count[line[1][j]]={}
                    self.pos_word_count[line[1][j]][line[0][j]]=1
        
        #### Get the initial state probabilties. This will be used in HMM and Viterbi 
        self.hmm_S0_prob = {}

        total = sum(self.hmm_S0_count.values())

        for each in self.hmm_S0_count.keys():
            self.hmm_S0_prob[each]=self.hmm_S0_count[each]/total

        #### Get the transition probabailities for the states. This will be used in HMM and Viterbi 
        self.hmm_Si_to_Sj_prob = {'adj':{}, 'adv': {}, 'adp':{}, 'conj':{}, 'det':{}, 'noun':{}, 'num':{}, 'pron':{},'prt':{}, 'verb':{}, 'x':{},'.':{} }

        for each in self.hmm_Si_to_Sj_count.keys():

            total = sum(self.hmm_Si_to_Sj_count[each].values())
    

            for pos in self.hmm_Si_to_Sj_count[each].keys():
                self.hmm_Si_to_Sj_prob[each][pos]=self.hmm_Si_to_Sj_count[each][pos]/total

        #Get the prior proabailites for the Parts of Speech:
        self.pos_prob={}

            ## Get the overall pos count 
        count=0

        
        for pos in self.pos_count.keys():
            count= count+self.pos_count[pos]
            
            ## Using the count obatined in the previous step, divide each value of self.pos_count dictionary to get the probability
        for pos in self.pos_count.keys():
            self.pos_prob[pos]=self.pos_count[pos]/count
       
        #Get the prior proabailites (Likelihood Values) for the Parts of Speech:
        self.pos_word_prob={'adj':{}, 'adv': {}, 'adp':{}, 'conj':{}, 'det':{}, 'noun':{}, 'num':{}, 'pron':{},'prt':{}, 'verb':{}, 'x':{},'.':{} }
       
            ## Using the count from self.pos_count dictionary we will get the likelihood values
        for pos in self.pos_word_count.keys():
            count= self.pos_count[pos]
           
            for word in self.pos_word_count[pos].keys():
                self.pos_word_prob[pos][word]=self.pos_word_count[pos][word]/count

        print(self.hmm_S0_prob)
        # print(self.hmm_Si_to_Sj_prob)
        for each in self.hmm_Si_to_Sj_count.keys():
            print( each," : ", self.hmm_Si_to_Sj_count[each])
        print("__"*50)
        for each in self.hmm_Si_to_Sj_prob.keys():
            print( each," : ", self.hmm_Si_to_Sj_prob[each])

        # print(self.pos_count)
        # print(self.pos_prob)
        # print("__"*50)

        # Validation 
        # Run the following code if you wan to check is the count of wrods in self.pos_count is equal to self.pos_word_count
        # for each in self.pos_word_count.keys():
        #     word_count=0
        #     for word in self.pos_word_count[each].keys():
        #         word_count=word_count+self.pos_word_count[each][word]
        #     print(each, ":", word_count)

        
            

    

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        

        # print(sentence)
        # Sequence of pos for a sentence
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

        # Using Viterbi algorithm to obtain the MAP ( Maximum A-Posteriori Probabilities) for the sentences. This will help us to get the most likely pos sequeneces given the observed words
     
        ## This involves 3 parts:
            #1) Get all the probailities
                # a) Transition probabilties (HMM State diagram)
                # b) Initial state probabilities 
                # c) Emisssion prababilties 
            #2) Get the Maximum A-Posteriori Probabilties using the Viterbi Algorithm 
            #3) Back track the states through which we were able to obtain this MAP, which will gives us the POS Sequence of the given words

        V_table = {}
        which_table= {'adj':{}, 'adv': {}, 'adp':{}, 'conj':{}, 'det':{}, 'noun':{}, 'num':{}, 'pron':{},'prt':{}, 'verb':{}, 'x':{},'.':{} }
        #Creating a Viterbi table 
        for s in self.hmm_S0_prob.keys():
            V_table[s]=[0]*len(sentence)
        
        ## First lets use the mutliplication of prababilities, Next lets use the logarthmic values 
        for s in self.hmm_S0_prob.keys():
            if sentence[0] in self.pos_word_prob[s].keys():
                V_table[s][0] =  self.hmm_S0_prob[s] * self.pos_word_prob[s][sentence[0]]
            else:
                 V_table[s][0] = 0
        # print(sentence)

        # for each in self.hmm_Si_to_Sj_prob.keys():
        #     print(each, len(self.hmm_Si_to_Sj_prob[each]))
        #     if len(self.hmm_Si_to_Sj_prob[each])!=12:
        #         print(self.hmm_Si_to_Sj_prob[each])


        for i in range(1,len(sentence)):
            # print("The value of i is ",i)
            for s in self.hmm_S0_prob.keys():
                # print("The value of s is ",s)
                
                (which_table[s][i], V_table[s][i]) =  max( [ (s0, V_table[s0][i-1] * self.hmm_Si_to_Sj_prob[s0][s]) for s0 in self.hmm_S0_prob.keys() if  s in self.hmm_Si_to_Sj_prob[s0].keys() ], key=lambda l:l[1] ) 
                if sentence[i] in self.pos_word_prob[s].keys():
                    V_table[s][i] *= self.pos_word_prob[s][sentence[i]]
                else:
                     V_table[s][i]=0


        # for  each in V_table:
        #     print(V_table[each])

        # print("Length of sentence ",len(sentence))
        # print(sentence)
        # print("WHich table ")
        # for each in which_table:
        #     print(each,":", which_table[each])

        # Here you'll have a loop that backtracks to find the most likely state sequence
        N= len(sentence)
        viterbi_seq = [""] * N

        ## The following for loop will the pos tagging for tha last word in sentence
        prob=0
        for s in self.hmm_S0_prob.keys():
            new_prob=V_table[s][N-1]
            if new_prob>=prob:
                viterbi_seq[N-1]=s
                prob=new_prob

        # 
        # print(which_table.keys())
        for i in range(N-2, -1, -1):
            viterbi_seq[i] = which_table[viterbi_seq[i+1]][i+1]
        print(viterbi_seq)

            
                
        # viterbi_seq[N-1] = "R" if V_table["R"][i] > V_table["S"][i] else "S"
        # for i in range(N-2, -1, -1):
        #     viterbi_seq[i] = which_table[viterbi_seq[i+1]][i+1]

        # print(25/0)


        return viterbi_seq

      


# Here you'll have a loop to build up the viterbi table, left to right
# for s in states:
#     V_table[s][0] = initial[s] * emission[s][observed[0]]

# for i in range(1, N):
#     for s in states:
#         (which_table[s][i], V_table[s][i]) =  max( [ (s0, V_table[s0][i-1] * trans[s0][s]) for s0 in states ], key=lambda l:l[1] ) 
#         V_table[s][i] *= emission[s][observed[i]]

# #       Easier to understand but longer version that does the same as the above two lines:
# #        V_table[s][i] = emission[s][observed[i]]
# #        if V_table["R"][i-1] * trans["R"][s] > V_table["S"][i-1] * V_table["S"][i-1] * trans["S"][s]:
# #            V_table[s][i] *= V_table["R"][i-1] * trans["R"][s]
# #            which_table[s][i] = "R"
# #        else:
# #            V_table[s][i] *= V_table["S"][i-1] * trans["S"][s]
# #            which_table[s][i] = "S"

# # Here you'll have a loop that backtracks to find the most likely state sequence
# viterbi_seq = [""] * N
# viterbi_seq[N-1] = "R" if V_table["R"][i] > V_table["S"][i] else "S"
# for i in range(N-2, -1, -1):
#     viterbi_seq[i] = which_table[viterbi_seq[i+1]][i+1]


        

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

