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
import copy


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

        # Creating a dictionary that has (parts of speech 1 and speech2)  as keys and values are again dictionaries (This dictionary has the words as the keys and the values will be their count of given word for the given parts of speech )
        self.pos1_pos2_word_count={}

        # Creating a dictionary to capture the initial state i.e. The POS of the first word in each of the sentence
        self.hmm_S0_count= {}

        # HMM: Creating a dictionary to capture the transition probilities for the state  (For HMM. Next state depends only on the current state )
        self.hmm_Si_to_Sj_count= {}

        #Complex: Creating a dictionary to capture the transition probabilities for the state ( For Complex model given: Next state depends not only on the current state and but also on the immediate past sate )
        self.complex_Si_Si_1_to_Sj_count={}

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
                ################################# Code for obataining the complex probabailities###################

                if j>=2:
                    _key= (line[1][j-2], line[1][j-1])
                    _value= line[1][j]


                    if _key in self.complex_Si_Si_1_to_Sj_count.keys():
                        if _value in self.complex_Si_Si_1_to_Sj_count[_key].keys():
                            self.complex_Si_Si_1_to_Sj_count[_key] [_value] +=1
                        else:
                            self.complex_Si_Si_1_to_Sj_count[_key] [_value] =1
                    else:
                        self.complex_Si_Si_1_to_Sj_count[_key]={}
                        self.complex_Si_Si_1_to_Sj_count[_key] [_value]= 1
               

                #############################################

                ## Updating Parts of Speech Count (P(S))
                if line[1][j] in self.pos_count.keys():
                    self.pos_count[line[1][j]] += 1
                else:
                    self.pos_count[line[1][j]] = 1
                
                ## Updating Parts of Speech Word Count ( P(W/S)) :: Used for HMM- Viterbi ##############

                if line[1][j] in self.pos_word_count.keys():
                    if line[0][j] in  self.pos_word_count[line[1][j]].keys():
                        self.pos_word_count[line[1][j]][line[0][j]]+=1
                    else:
                        self.pos_word_count[line[1][j]][line[0][j]]=1
                else:
                    self.pos_word_count[line[1][j]]={}
                    self.pos_word_count[line[1][j]][line[0][j]]=1

                ## Updating Parts of Speech Word Count ( P(W/(S1 and S2))) :: Used for Complex Model ##############
                if j>=1:
                    _pos_key=(line[1][j-1],line[1][j])
                    _word = line[0][j]
                    if _pos_key in self.pos1_pos2_word_count.keys():
                        if _word in  self.pos1_pos2_word_count[_pos_key].keys():
                            self.pos1_pos2_word_count[_pos_key][_word]+=1
                        else:
                            self.pos1_pos2_word_count[_pos_key][_word]=1
                    else:
                        self.pos1_pos2_word_count[_pos_key]={}
                        self.pos1_pos2_word_count[_pos_key][_word]=1
            
        #### Get the initial state probabilties. This will be used in HMM and Viterbi 
        self.hmm_S0_prob = {}

        total = sum(self.hmm_S0_count.values())

        for each in self.hmm_S0_count.keys():
            self.hmm_S0_prob[each]=self.hmm_S0_count[each]/total
        ##########################################################################################################
        #### Get the transition probabailities for the states of HMM model. This will be used in HMM and Viterbi 
        self.hmm_Si_to_Sj_prob = {'adj':{}, 'adv': {}, 'adp':{}, 'conj':{}, 'det':{}, 'noun':{}, 'num':{}, 'pron':{},'prt':{}, 'verb':{}, 'x':{},'.':{} }

        for each in self.hmm_Si_to_Sj_count.keys():

            total = sum(self.hmm_Si_to_Sj_count[each].values())
    

            for pos in self.hmm_Si_to_Sj_count[each].keys():
                self.hmm_Si_to_Sj_prob[each][pos]=self.hmm_Si_to_Sj_count[each][pos]/total
        ###############################################################################################################
        #### Get the transition probabilties for the states of Complex Model. This will be used in the Complex model (Gibbs Sampling)

        self.complex_Si_Si_1_to_Sj_prob= copy.deepcopy( self.complex_Si_Si_1_to_Sj_count)

        for each in self.complex_Si_Si_1_to_Sj_count.keys():

            total = sum(self.complex_Si_Si_1_to_Sj_count[each].values())

            for pos in self.complex_Si_Si_1_to_Sj_count[each].keys():
                self.complex_Si_Si_1_to_Sj_prob[each][pos]=self.complex_Si_Si_1_to_Sj_count[each][pos]/total

        #####################################################################################################################


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
       
        ############### Using the count from self.pos_count dictionary we will get the likelihood values
        for pos in self.pos_word_count.keys():
            count= self.pos_count[pos]
           
            for word in self.pos_word_count[pos].keys():
                self.pos_word_prob[pos][word]=self.pos_word_count[pos][word]/count

        ##############
       

        self.pos1_pos2_word_prob= copy.deepcopy( self.pos1_pos2_word_count)

        for each in self.pos1_pos2_word_count.keys():

            total = sum(self.pos1_pos2_word_count[each].values())

            for word in self.pos1_pos2_word_count[each].keys():
                self.pos1_pos2_word_prob[each][word]=self.pos1_pos2_word_count[each][word]/total

        print(self.hmm_S0_prob)
        # print(self.hmm_Si_to_Sj_prob)
        # for each in self.hmm_Si_to_Sj_count.keys():
        #     print( each," : ", self.hmm_Si_to_Sj_count[each])
        # print("__"*50)
        # for each in self.hmm_Si_to_Sj_prob.keys():
        #     print( each," : ", self.hmm_Si_to_Sj_prob[each])

        # print(self.pos_count)
        # print(self.pos_prob)
        # print("__"*50)
        for each in self.complex_Si_Si_1_to_Sj_count:
            print(each , "::", self.complex_Si_Si_1_to_Sj_count[each])
            print(each , "::", self.complex_Si_Si_1_to_Sj_prob[each])
        # # Validate the counts in complex transition dictionary
        # validate_count=0
        # for each in self.complex_Si_Si_1_to_Sj_count:
        #     for i in self.complex_Si_Si_1_to_Sj_count[each]:
        #         validate_count=validate_count+self.complex_Si_Si_1_to_Sj_count[each][i]
        # print(validate_count)

       
        # Validation 
        # Run the following code if you wan to check is the count of wrods in self.pos_count is equal to self.pos_word_count
        # for each in self.pos_word_count.keys():
        #     word_count=0
        #     for word in self.pos_word_count[each].keys():
        #         word_count=word_count+self.pos_word_count[each][word]
            # print(each, ":", word_count)
        # print(25/0)

        
            

    

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
                        word_prob[pos]=0.0000000001
            # print("The word is ", word)
            # print(word_prob)
            max_prob=0.0000000001
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
                # print("used log1")
                V_table[s][0] = - math.log( self.hmm_S0_prob[s] * self.pos_word_prob[s][sentence[0]] )
            else:
                 V_table[s][0] = -math.log(0.000000000001)
        # print(sentence)

        # for each in self.hmm_Si_to_Sj_prob.keys():
        #     print(each, len(self.hmm_Si_to_Sj_prob[each]))
        #     if len(self.hmm_Si_to_Sj_prob[each])!=12:
        #         print(self.hmm_Si_to_Sj_prob[each])


        for i in range(1,len(sentence)):
            # print("The value of i is ",i)
            for s in self.hmm_S0_prob.keys():
                # print("The value of s is ",s)
                # print("used log2")
                (which_table[s][i], V_table[s][i]) =  min( [ (s0, V_table[s0][i-1] -math.log( self.hmm_Si_to_Sj_prob[s0][s])) for s0 in self.hmm_S0_prob.keys() if  s in self.hmm_Si_to_Sj_prob[s0].keys() ], key=lambda l:l[1] ) 
                if sentence[i] in self.pos_word_prob[s].keys():
                    V_table[s][i] = V_table[s][i]-math.log( self.pos_word_prob[s][sentence[i]])
                else:
                    V_table[s][i]=V_table[s][i]-math.log(0.000000000001)


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
        prob=V_table['noun'][N-1] ### You can take anything, I took noun
        for s in self.hmm_S0_prob.keys():
            new_prob=V_table[s][N-1]
            if new_prob<=prob:
                viterbi_seq[N-1]=s
                prob=new_prob

        # 
        # print(which_table.keys())
        for i in range(N-2, -1, -1):
            viterbi_seq[i] = which_table[viterbi_seq[i+1]][i+1]
        # print(viterbi_seq)

            
                
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

        ## Lets generate large number of samples or particles (may be in thousands) for each sentence and once we generate them lets calculate the probability foe each of the words

        sample_count= 50
        
        ## Lets create a dictionary of the dimensions that stores the pos values generated by each of the particle
        samples= [["" for __ in range(len(sentence)) ] for  _ in range(sample_count)]
        # print(" Size of samples is {} * {}".format(len(samples),len(samples[0])))
        # print(samples)
      
        all_pos_values= ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron','prt', 'verb', 'x','.' ]

        ## Randomly assign values for the first sample and from second samples onwards our logic will fill in the values

        # for i in range(len(sentence)):
        #     samples[0][i]=random.choice(all_pos_values)
        samples[0]=self.simplified(sentence)
        
        # print(25/0)

        for itr in range(1,sample_count):
            samples[itr]=copy.deepcopy(samples[itr-1] )## Assign the pos values that are obtained in the previous iteration 

            for k in range(len(sentence)):

                temp_pos_values= {'adj':{}, 'adv': {}, 'adp':{}, 'conj':{}, 'det':{}, 'noun':{}, 'num':{}, 'pron':{},'prt':{}, 'verb':{}, 'x':{},'.':{} }
                ## The aim of this of this for loop is that for each of the word we assume that the the other pos values are fixed and obtain the probability distribution for that word
                ##### Once we have the probabailty distribution then we try to flip a coin to get the parts of speech.


                ### To get the distribution we have find proababiity for all pos values, Only once we have that then we can toss a coin or do some random process to get the probabality
                for each_pos in temp_pos_values.keys():

                    parts_of_speech= copy.deepcopy(samples[itr])
                    parts_of_speech[k]=each_pos

                    prob_dist=1
                    for pos_position in range(len(sentence)):

                        if pos_position==0:
                            P=parts_of_speech[pos_position]
                            W=sentence[pos_position]
                            try:
                                prob_dist=prob_dist* self.hmm_S0_prob[P] * self.pos_word_prob[P][W]
                            except KeyError:
                                prob_dist=prob_dist*0
                            
                        elif pos_position==1:
                            Pi=parts_of_speech[pos_position-1]
                            Pj=parts_of_speech[pos_position]
                            W=sentence[pos_position]

                            try:
                                prob_dist=prob_dist * self.hmm_Si_to_Sj_prob[Pi][Pj] * self.pos1_pos2_word_prob[(Pi,Pj)][W]
                            except KeyError:
                                prob_dist=prob_dist*0
                                
                        else:
                            Pj=parts_of_speech[pos_position]
                            Pi=parts_of_speech[pos_position-1]
                            Pi_1=parts_of_speech[pos_position-2]
                            W=sentence[pos_position]

                            try:
                                prob_dist=prob_dist* self.complex_Si_Si_1_to_Sj_prob[(Pi_1,Pi)][Pj] * self.pos1_pos2_word_prob[(Pi,Pj)][W]
                            except:
                                prob_dist=prob_dist*0
                                
                    temp_pos_values[each_pos]=prob_dist


                total_prob=sum(temp_pos_values.values())

                for each in temp_pos_values.keys():
                    try:
                        temp_pos_values[each] = temp_pos_values[each]/(total_prob)
                    except ZeroDivisionError:
                        temp_pos_values[each] = temp_pos_values[each]/1
                # print(temp_pos_values)
                ### Get the range of proabbilty values that our parts of speech labels can take. Once we have the range then we can generate the random number between 0 and 1 and see the range it falls in range and take that pos
                pos_range=copy.deepcopy(temp_pos_values)
                prob_tracker=0
                for each in temp_pos_values.keys():
                    pos_range[each]=(prob_tracker,prob_tracker+temp_pos_values[each])
                    prob_tracker=prob_tracker+temp_pos_values[each]

                # print(pos_range)
                
                rand_value= random.random()
                # print(rand_value)
                for each in pos_range.keys():
                    if rand_value >=pos_range[each][0] and rand_value<=pos_range[each][1]:
                        samples[itr][k]=each
                        break
                        # samples[itr][k]=max(temp_pos_values, key=temp_pos_values.get)
       
        pos_final=[]
        for each in range(len(sentence)):
            # temp_pos_values= {'adj':{}, 'adv': {}, 'adp':{}, 'conj':{}, 'det':{}, 'noun':{}, 'num':{}, 'pron':{},'prt':{}, 'verb':{}, 'x':{},'.':{} }
            lst=[x[each] for x in samples]
            pos_final.append(max(lst,key=lst.count))

                # print("Pradeep Reddy Rokkam")



        return pos_final



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

