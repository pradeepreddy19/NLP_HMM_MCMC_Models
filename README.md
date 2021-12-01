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

NOTE: In the formula to get the posterior probabailty, we ignored the denominator as it is same for all the poslables and we are only interested in getting the pos that has the maximum value 

Observations:
* Even though the logic it works extremely well when it comes to the accuracy of the words
  The acccuacy on the the testing set is almost 91
  


### HMM - Solved Using Viterbi Algorithm:

Problems:
* We faced the issue of probabailties becoming very small and due this we were not able to procedd further and the accuracies for the words and sentences was very low 
 *  Resolution: The way how to tackled this problem of small probabalities is to use lograthimc value of those  probabilties 


### MCMC - Solved using Gibbs Sampling:

Pointers:
75 samples
taking lot of time for 1000 samples 

Note: All the probabilties here are only for numerators and 
