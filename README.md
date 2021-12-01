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


# **Part -3: READING TEXT**

**Aim** : To recognize text from noisy images of English sentences.

## **General approach: -**

In this problem, we were to use the provided courier-train.png picture as a reference image for all characters (capital and small) along with punctuations to determine the actual difference between this and the noisy image data. In order to predict each character in a given word, we can model it into a HMM problem where the hidden variables are the ones we want to recognize and the observed variables are the sub-image corresponding to each alphabet. We understood the versatility of HMMs through it&#39;s wide area of applications from POS tagging to recognizing text. Here, we can also classify using a simple Naïve Bayes Classifier.

## **Challenges &amp; Implementing requirements: -**

The main challenge was to figure out how to calculate the emission probabilities. One obvious way was to compare the noisy and reference image alphabet-by-alphabet and store the differences in pixels. At first, we tried implementing with the count of pixels that match. In this scenario, we observed that the model predicted the alphabet &quot;e&quot; more often than any other alphabet since this alphabet in general has a high occurrence in the English language and the respective prior probability was dominating over the other values.

We then incorporated the count of matching stars, blank spaces and the count of pixels that did not match. Since each of these have different occurrences, we decided to give each of it a weight corresponding to its frequency of occurrence, i.e. :

**" matching stars > matching blank spaces > pixels that do not match " (decreasing frequency of occurrence)**

Since the prior probabilities P(Observed Letter) were too dominating, we decided to experiment by considering the likelihood values P (Hidden Letter | Observed Letter) alone to judge which alphabet is most likely to take the place in each word in each sentence.

## **Code &amp; Solution explanation: -**

**Simple Method:**

As explained above, we took a weighted counts of each of matching stars, matching blank spaces and pixels that don&#39;t match and divided it with the total number of pixels in that character (14\*25). We traversed through each character in every word of a sentence in a test image, compared it with the respective reference image and calculated the probability of each character in that position. We then select the one with the highest probability (likelihood in our case) as the final decision and append it to one final string.

**HMM Viterbi:**

We used the train.txt file from part 1 after removing the POS tags to train and retrieve the initial and transition probabilities. For combinations of transitions not observed, we allocated a small value of 10^-10 in order to avoid KeyErrors and to consider every possibility.

In order to calculate emission probabilities, we used the same approach as mentioned in the Simple Method above that utilizes Naïve Bayes for calculation.

We take positive log of the probabilities and take max of the computed values and store it into the Viterbi table. We then use backtracking to put together the final string based on highest likelihoods.

## **Observations and Inferences: -**

As mentioned earlier, we observed that the alphabet &quot;e&quot; was predicted more often while taking prior probabilities into consideration as they were more domination than the likelihood values. The results were better after we considered highest likelihood alone in both the simple naïve bayes as well as HMM Viterbi.

The results were further improved after weights were introduced into the picture. As we changed the weights, we could fine tune the model to perform better. We noticed that &quot;(&quot; would be repeated more often in the place of &quot;i&quot; and spaces. As we fine tuned the weights, the HMM Viterbi model outperformed the simple naïve bayes model. In some cases we could tell that the transition probabilities made a huge difference as we noticed a &quot;b&quot; in front of the word &quot;right&quot; since the word &quot;bright&quot; could have been a common occurrence. &quot;The&quot; would also often become &quot;Them&quot; since the model has knowledge of one state before.

This knowledge of the previous state in HMM Viterbi sometimes ended in a better result than simple naïve bayes and sometimes gave a less accurate result as well.
