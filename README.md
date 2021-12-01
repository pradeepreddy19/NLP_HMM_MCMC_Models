# harmohan-jzayatz-prokkam-a3
a3 created for harmohan-jzayatz-prokkam

## Part 1: Part-of-speech tagging:

<b>Aim:</b> Given a new sentence, identify its parts of speech tagging
       

### Simple Model:

Training:
* Need the probabalities of pos 
  *  Get the frequency of occurance for each and then divide it by the total count to get the pra=obabality for each pos
    * eg: p(noun), p(adv) etc.,
* Need Likelihood values or emission probabailities:
  * For a given pos, what is the probabailty of the given word
    * Get the frequencey of words for a given pos and then divide it by the total count of words that are in the given pos
      * eg: p(is/pron), p(eai/noun) etc.,
     
Algorithm:
The implementation of this algorithm is very simple as the logic is very simple, we are only using simple Baye's rule to get the pos tagging for any given word.
P(pos/word) = { p(word/pos) * p(pos) } / p(word)

For any given word, we apply the the above formula for all the pos labels and take the one which is having a maxium value. 

NOTE: In the formula to get the posterior probabailty, we ignored the denominator as it is same for all the pos labels and we are only interested in getting the pos that has the maximum value 

Running time: Th execution time for the code to test the given 2000 sentences is under one 1 minute 

Observations:
* Even though the logic is very simple it works extremely well when it comes to the accuracy of the words and does pretty well in identifying the entire sentence as well
       * The acccuacy for the words on the the testing set is almost 91.76% and the accuracy for the sentences is 37%
     

### HMM - Solved Using Viterbi Algorithm:

Training:
 Along the with the training variables that were used in the Simple model we also need couple of other variables to solve the HMM. The variables that were required are called inital state probabailties and transistion proababities. These are the the probabilties that define the state trasition diagram and these probabilties are useful in the solving the Markov chain which is a special case of the Bayes Net 
 
 Initial State Proababilties:
 * What is the probabaility that a sentence starts with a particular pos
 * eg: P(S0=Noun), P(S0=Adv) etc.,

Transition probabailties:
* What is the proababitiy of trasnition from noun to verb?
* P(si/Sj), P(Si=Noun/ Sj= Adv) etc.,

Running Time:Just like the simple model the execution time for the code to test the given 2000 sentences is under one 1 minute. Ofcourse it takes slightly more time than the simple model 

Problems:
* We faced the issue of probabailties becoming very small and due this we were not able to procedd further and the accuracies for the words and sentences was very low 
 *  Resolution: The way we tackled this problem of small probabalities is to use lograthimc value of those probabilties ( similar methodlogy that was dicussed in the class)


### MCMC - Solved using Gibbs Sampling:

Running Time:Running time is one of the major issues in MCMC implementation. As per the time requirement mentioned in the assignment (code should run under 10 minutes) we were able to run only 75 samples  for each of the sentences. 

Results :
Pointers:
75 samples
taking lot of time for 1000 samples 

Note: All the probabilties here are only for numerators and 
