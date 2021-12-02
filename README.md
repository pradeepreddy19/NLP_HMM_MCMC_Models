# harmohan-jzayatz-prokkam-a3
a3 created for harmohan-jzayatz-prokkam


## Part 1: Part-of-speech tagging:



<b>Aim:</b> Given a new sentence, identify its parts of speech tagging


## **General approach: -**
In this problem, we were to use the provided bc.train to get all pos taggings for the words  and estimate the required prior probabailties, likelihoods and when a new sentence is provided from the test dataset we have identify its pos taggig for the words along with with log  of posterior probabailties. In order to predict pos tagging for each given word in the sentence, we can model it in three different ways as recommended by the Professor. 
 * We can use a simple model where we calculate the posteroir probabailty P(pos/word) for all the pos and take the pos that has the maximum posterior probabailty value  
 * We can also use a HMM model where the hidden variables (pos tagging) are the ones we want to recognize and the observed variables are the words provided in the sentence
 * We can also use a complex model, but this a Bayes Net and not a Markov chain,so we cannot apply Viterbi algorithm and in general if we want to get the correct answer for this problem it is NP hard as takes way too much time to solve the problem. However, instead we shift our focus to an approximate solution we can solve this problem in reasonable amount of time and this is acheived through one of the techniques called Gibb's sampling

### Simple Model:

**Algorithm:**
The implementation of this algorithm is very simple as the logic is very simple as we are only using simple Baye's rule to get the pos tagging for any given word.
**P(pos/word) = { p(word/pos) * p(pos) } / p(word)**

For any given word, we apply the the above formula for all the pos labels and take the one which is having a maximuum value given by the formula. 

**NOTE:** In the formula to get the posterior probabailty, we ignored the denominator as it is same for all the pos labels as we are only interested in getting the pos that has the maximum value 

To estimate the posterior proababilty for any word  p(pos/word) we need likelihood values and prior probabilities and that is obtained in the training phase

**Training:**
* *Proir probabalities of pos:* 
  *  Get the frequency of occurance for each pos and then divide it by the total count to get the probabality for each pos
    * eg: p(noun), p(adv) etc.,
* *Likelihood values or emission probabailities:*
  * For a given pos, what is the probabailty of the given word
    * Get the frequencey of words for a given pos and then divide it by the total count of words that are in the given pos
      * eg: p(is/pron), p(eai/noun) etc.,
     

**Running time:** The execution time for the code to test the given 2000 sentences is under one 1 minute for the simple model

**Results**: The acccuacy for the words on the the testing set is almost **91.76%** and the accuracy for the sentences is **37.75%**

**Observations:**
* Even though the logic is very simple it works extremely well when it comes to the accuracy of the words and does pretty well in identifying the entire sentence as well
       
     
### HMM - Solved Using Viterbi Algorithm:

**Algorithm**: We used Viterbi Algorthim to solve the HMM. For any given sentence we apply Viterbi Algorithm


**Training:**
Along the with the training variables that were used in the Simple model we also need couple of other variables to solve the HMM. The variables that were required are called inital state probabailties and transistion proababities. These are the the probabilties that define the state trasition diagram and these probabilties are useful in the solving the Markov chain which is a special case of the Bayes Net 
 
 Initial State Proababilties:
 * What is the probabaility that a sentence starts with a particular pos?
 * eg: P(S0=Noun), P(S0=Adv) etc.,

Transition probabailties:
* What is the proababitiy of trasnition from noun to verb?
* P(si/Sj), P(Si=Noun/ Sj= Adv) etc.,

**Running Time:** Just like the simple model the execution time for the code to test the given 2000 sentences is under one 1 minute. Ofcourse it takes slightly more time than the simple model but still runs very quickly and takes time that is under 60 seconds


**Results**: The acccuacy for the words on the the testing set is almost **95.04%** and the accuracy for the sentences is **54.20%**

**Observations:** HMM does a great job when it comes to accuracy of the words and and the senetnces. This is a great improvement from the the simple model. The running time of the algorithm is under a minute and this again shows that this special case of Bayes Net is very important in solving the problem.

**Problems:**
* We faced the issue of probabailties becoming very small and due this we were not able to proceed further and the accuracies for the words and sentences was very low 
 * ***Resolution***: The way we tackled this problem of small probabalities is to use lograthimc value of those probabilties ( similar methodlogy that was dicussed in the class)


### MCMC - Solved using Gibbs Sampling:

**Algorithm:**
The algorithm that we used to implement the Complex model was using Gibb's sampling. This particular algorithm is little bit complex when compared to the other two models
 * **Gibss Sampling:**
  * Initial sample was obtined by taking a random pos for each of the words in a sentence and started performimg Gibb's sampling. We can also the pos tagging that was obtained from the simple model
  * Repeat: (#  Sample count)
    * Copy the previous sample to the current sample
    * Repeat (#Words in a the given sentence)
      * Find p(pos/word) for each word and assuming all the other words pos is given ( we can get this information from the prvious sample)
      * The above step gives the probabailty distribution for all the pos
      * Now toss a #pos sided coin and with probabilty distribution obtained in the previous step
      * Take the side that has come and assign that pos to the word
 * Once we have the samples that have all the pos taggings, we can now calculate the pos tagging of each word by simply calculating the proabailty for all pos and take the pos that has the maximum value

**Training:**
Along the with the training variables that were used in the Simple model and HMM model we also need two more variables toomplement the complex variable.
 
Emission Proababilties:
 * What is the probabaility that a word gives pos1 and pos0?
 * eg: P(W/Si,Sj), P(eai/(Noun,Adv), P(Indiana/(Noun,Conj) etc.,

Complex Transition probabailties:
* What is the proababitiy of transition from noun,conj to verb?
* P(Si/(Si-1,Si-2), P(Si=Verb/ Si-1=Conj,Si-2=Noun) etc.,


**Running Time:** Running time is one of the major issues in MCMC implementation. As per the time requirement mentioned in the assignment (code should run under 10 minutes) we were able to run only 75 samples  for each of the sentences. 


**Results:** The acccuacy for the words on the the testing set is almost **92.39%** and the accuracy for the sentences is **43.95%**

**Observations**: The model performs better than the simple model but not better than HMM. It could be beacuse of the time constraint of 10 minutes as we were only able to get 75 samples for each of the sentence. We could have obatined more acuuracies if we had taken more samples like in thousands for each sentence as we know that MCMC approximately converges to Bayes Net given in the complex model. If the actual result of the complex model (which is a Bayes Net) gives a resul of 100% then by taking more samples (tending to inifiinity) in Gibb's sampling would give us the result approximately close to 100%

**NOTE**: We may get different accuracies for words and sentences for every run (only for the Complex model). The reason we may get such differences is that we are performing a random experiment once we obtain the probabilty distribution. However, the varince is not huge atleast for this training and test data. The range is between **93%** and **94%** for the words and **43%** to **45%** for the sentences

**Screenshot of the Results:**
 <img width="1440" alt="image" src="https://media.github.iu.edu/user/18258/files/571b8180-52c8-11ec-881e-e96dcfec15d3">


## Part 2:  Ice Tracking
Note:  I had to run `pip install imageio` on SILO in order to run polar.py
 
## Objective and formulation
The objective of this problem is to inference on a Bayesnet to estimate where the air/ice and rock/ice boundary layers are located in a given image. The approach uses probabilistic models to determine which pixels are more likely to be a boundary or not.
Three model types used are:
1. Simple model:  probability of a hidden state is determined by emission probability only
2. HMM:  probability of s hidden state is determined by probability of previous hidden state, transition probability and emission probability.
3. HMM with human feedback:  coordinate indicated by user input to assumed to have probability of 1 of being a boundary and probability of 0 of being not a boundary
 
## Description of how program works
 
The conditional probabilities required for the above described models include the following probability distributions (B = boundary, nB = not boundary):
* initial probabilities:  P(B) = 0.5, P(nB) = 0.5.  To give equal probability that a pixel could be a boundary or not.
* transition probabilities:  P(BB) = 0.9, P(nBB) = 0.1, P(nBB) = 0.1, P(nBnB) = 0.9.  To encourage smoothness of the boundary.  
* emission probabilities: Are based on the value of the edge_strength mask.  Edge Strength values are binned into 9 bins and then assigned a probability based on bin number.  P(bin|B) = [.01,.02,.03,.04,.05,.10,.15,.20,.40].  P(bin|nB)= [.88,.05,.01,.01,.01,.01,.01,.01,.01].    
 
These values are converted to log probabilities so computation of conditional probabilities are calculated with summations instead of products.
 
For the __simple model__, non_boundary state emission probabilities are subtracted from boundary state emission probabilities and the location where the difference is greatest is chosen as the location of the boundary.
 
The __HMM model__ uses the same comparison of probabilities, but compares the conditional probability of the previous state, transition probability, and emission probability.  The Viterbi algorithm is used so only the maximum probability of each hidden state is retained and used to calculate the conditional probabilities of the next state.
 
The __feedback model__ uses the same HMM/Viterbi algorithm but initially shifts the image left so that the column where the feedback coordinate is becomes the first column.  The initial and emission probabilities for the row indicated by the user is set to 1 for the boundary state probability distribution and 0 (it's actually set to a very small number so log of 0 does not occur) for the non_boundary state probability distribution.  This should make the model start the boundary at the user feedback coordinates and procede from there.  The model generated boundary is then shifted right to return it to its original position.
 
## Discussion (assumptions, results, challenges)
Assumptions:  
* The air/ice boundary is always above the ice/rock boundary.  The buffer we chose to use was 30 pixels in order to account for the thickness of the air/ice boundary.  
* each boundary layer extends the width of the image
* the higher the edge_strength value, the higher the probability that that pixel is part of a boundary
* The noise in the image occurs mostly in non_boundary states and have lower edge_strength values.  
 
With these assumptions and the models described above, our model produced the following boundaries for the test images. 
 
<img width="1440" alt="Boundaries drawn on Test Images" src="https://github.iu.edu/cs-b551-fa2021/harmohan-jzayatz-prokkam-a3/blob/202f887cdf74eb4d1135229d612793687bebee1b/part2/collage.png">
 
Results:  
The model was able to generate smooth, reasonable boundaries for some parts of some images.  It performed better on air/ice boundaries since those are more distinct and have less vertical variation. Ice/rock layers in images 16.png and 23.png were not as well generated.  It also appears that some pixels are in unexpected locations.  
 
Challenges:     
The results show that the models could be improved to generate smoother boundaries and be able to discriminate boundaries that go through regions of noise that occur as what looks like vertical smudges that decrease the edge_strengh values.  


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

We used the train.txt file from part 1 after removing the POS tags to train and retrieve the initial and transition probabilities. For combinations of transitions not observed, we allocated a small value of 1^-10 in order to avoid KeyErrors and to consider every possibility.

In order to calculate emission probabilities, we used the same approach as mentioned in the Simple Method above that utilizes Naïve Bayes for calculation.

We take positive log of the probabilities and take max of the computed values and store it into the Viterbi table. We then use backtracking to put together the final string based on highest likelihoods.

## **Observations and Inferences: -**

As mentioned earlier, we observed that the alphabet &quot;e&quot; was predicted more often while taking prior probabilities into consideration as they were more domination than the likelihood values. The results were better after we considered highest likelihood alone in both the simple naïve bayes as well as HMM Viterbi.

The results were further improved after weights were introduced into the picture. As we changed the weights, we could fine tune the model to perform better. We noticed that &quot;(&quot; would be repeated more often in the place of &quot;i&quot; and spaces. As we fine tuned the weights, the HMM Viterbi model outperformed the simple naïve bayes model. In some cases we could tell that the transition probabilities made a huge difference as we noticed a &quot;b&quot; in front of the word &quot;right&quot; since the word &quot;bright&quot; could have been a common occurrence. &quot;The&quot; would also often become &quot;Them&quot; since the model has knowledge of one state before.

This knowledge of the previous state in HMM Viterbi sometimes ended in a better result than simple naïve bayes and sometimes gave a less accurate result as well.
