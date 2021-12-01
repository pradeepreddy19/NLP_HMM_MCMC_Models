# harmohan-jzayatz-prokkam-a3
a3 created for harmohan-jzayatz-prokkam

## Part 1: Part-of-speech tagging:

<b>Aim:</b> Given a new sentence, identify its parts of speech tagging

Training:
* Need the probabalities of pos 
  *  Get the frequency of occurance for each and then divide it by the total count to get the pra=obabality for each pos
    * eg: p(noun), p(adv) etc.,
* Need Likelihood values:
  * For a given pos, what is the probabailty of the given word
    * Get the frequencey of words for a given pos and then divide it by the total count of words that are in the given pos
      * eg: p(is/pron), p(eai/noun) etc.,
       

### Simple Model:

Training:
  * Likelihood values:
The implementation of this algorithm is very simple as the logic is very simple, we are only using simple Baye's rule to get the pos tagging for any given word

For all the words in all the given sentences our aim to 


### HMM - Solved Using Viterbi Algorithm:

Problems:
* We faced the issue of probabailties becoming very small and due this we were not able to procedd further and the accuracies for the words and sentences was very low 
 *  Resolution: The way how to tackled this problem of small probabalities is to use lograthimc value of those  probabilties 


### MCMC - Solved using Gibbs Sampling:

Pointers:
75 samples
taking lot of time for 1000 samples 

Note: All the probabilties here are only for numerators and 
