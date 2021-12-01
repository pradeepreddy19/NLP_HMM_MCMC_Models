#!/usr/local/bin/python3
#
# Authors: Harini Mohanasundaram (harmohan), Pradeep Reddy Rokkam (prokkam), Joy Zayatz(jzayatz)
#
# Ice layer finder
# Based on skeleton code by D. Crandall, November 2021
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio
import numpy as np


# calculate "Edge strength map" of an image                                                                                                                                      
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_boundary(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

def draw_asterisk(image, pt, color, thickness):
    for (x, y) in [ (pt[0]+dx, pt[1]+dy) for dx in range(-3, 4) for dy in range(-2, 3) if dx == 0 or dy == 0 or abs(dx) == abs(dy) ]:
        if 0 <= x < image.size[0] and 0 <= y < image.size[1]:
            image.putpixel((x, y), color)
    return image


# Save an image that superimposes three lines (simple, hmm, feedback) in three different colors 
# (yellow, blue, red) to the filename
def write_output_image(filename, image, simple, hmm, feedback, feedback_pt):
    new_image = draw_boundary(image, simple, (255, 255, 0), 2)
    new_image = draw_boundary(new_image, hmm, (0, 0, 255), 2)
    new_image = draw_boundary(new_image, feedback, (255, 0, 0), 2)
    new_image = draw_asterisk(new_image, feedback_pt, (255, 0, 0), 2)
    imageio.imwrite(filename, new_image)

def bin_probabilites(hidden_state, observation):
    return emission_probs[hidden_state][observation]

def viterbi(kind, buffer, airice_layer, obs_p_b, obs_p_notb, initial_probs, transition_probs):
    if kind == 'airice':
        airice_layer = np.repeat(0,225)
        buffer = 0
    if kind == 'icerock':
        airice_layer = airice_layer
        buffer = buffer

    boundary = []

    B = []
    nB = []

    #different calculation for first column
    e_B = obs_p_b[:,0]
    e_nB = obs_p_notb[:,0]

    cond_B = initial_probs['B']+ e_B
    cond_nB = initial_probs['nB']+ e_nB

    B.append(cond_B)
    nB.append(cond_nB)

    compare = np.subtract(cond_B[int(airice_layer[0])+buffer:], cond_nB[int(airice_layer[0])+buffer:])

    boundary.append(argmax(compare)+int(airice_layer[0])+buffer)

    #viterbi for remaining columns
    for n in range(1, image_array.shape[1]):

        obs = obs_seq[:,n]
        e_B = obs_p_b[:,n]
        e_nB = obs_p_notb[:,n]
        previous_B = B[n-1]
        previous_nB = nB[n-1]

        B_from_B = previous_B + transition_probs['BB'] + e_B
        B_from_nB = previous_nB + transition_probs['nBB'] + e_B
        max_B = np.max(list(zip(B_from_B, B_from_nB)), axis = 1)
        B.append(max_B)

        nB_from_B = previous_B + transition_probs['BnB'] + e_nB
        nB_from_nB = previous_nB + transition_probs['nBnB'] + e_nB
        max_nB = np.max(list(zip(nB_from_B, nB_from_nB)), axis = 1)
        nB.append(max_nB)


        compare =np.subtract(max_B[int(airice_layer[n])+buffer:], max_nB[int(airice_layer[n])+buffer:])
        boundary = np.append(boundary,(argmax(compare)+int(airice_layer[0])+buffer))

    return boundary



# main program
#
if __name__ == "__main__":

    if len(sys.argv) != 6:
        raise Exception("Program needs 5 parameters: input_file airice_row_coord airice_col_coord icerock_row_coord icerock_col_coord")

    input_filename = sys.argv[1]
    gt_airice = [ int(i) for i in sys.argv[2:4] ]
    gt_icerock = [ int(i) for i in sys.argv[4:6] ]

    # load in image 
    input_image = Image.open(input_filename).convert('RGB')
    image_array = array(input_image.convert('L'))

    # compute edge strength mask -- in case it's helpful. Feel free to use this.
    edge_strength = edge_strength(input_image)
    imageio.imwrite('edges.png', uint8(255 * edge_strength / (amax(edge_strength))))

    
    
    #bin edge mask values and assign emission probabilities
    hist, bins = np.histogram(edge_mask, bins = [0,25,50,75,100,150,200,250,400])
    obs_seq = np.digitize(edge_mask, bins)


    e_p_bins = list(range(1,len(bins)+1))
    e_p_boundary = np.log([.01,.02,.03,.04,.05,.10,.15,.20,.40])
    e_p_not_boundary = np.log([.88,.05,.01,.01,.01,.01,.01,.01,.01])
    emission_probs = {'B':dict(zip(e_p_bins, e_p_boundary)),
                  'nB':dict(zip(e_p_bins, e_p_not_boundary))}

    vfunc = np.vectorize(bin_probabilites)

    obs_p_b = vfunc('B', obs_seq)
    obs_p_notb = vfunc('nB', obs_seq)
    
    
    # set initial probabilites
    hidden_states = ['B', 'nB']
    initial_p = np.log([.5,.5])
    initial_probs = dict(zip(hidden_states, initial_p))
    
    
    # set transition probabilites
    transitions = ['BB', 'BnB', 'nBB', 'nBnB']
    trans_p = np.log([.9,.1,.1,.9])
    transition_probs = dict(zip(transitions, trans_p))

    
    # air ice layers

    airice_simple = []
    for m in range(image_array.shape[1]):
        compare_p = np.subtract(obs_p_b[:,m], obs_p_notb[:,m])
        airice_simple = np.append(airice_simple,argmax(compare_p))


    airice_hmm = viterbi('airice', 0, 'na',obs_p_b, obs_p_notb, initial_probs, transition_probs)   


    obs_p_b_f= np.roll(obs_p_b, -(gt_airice[1]-1), axis =1)
    obs_p_notb_f = np.roll(obs_p_notb, -(gt_airice[1]-1), axis =1)
    obs_p_b_f[gt_airice[1], gt_airice[1]]=np.log(1)
    obs_p_notb_f[gt_airice[1], gt_airice[1]]=np.log(1e-20)


    initial_p_f = [np.repeat(np.log(.5), 175),np.repeat(np.log(.5), 175)]
    initial_probs_f = dict(zip(hidden_states, initial_p_f))
    initial_probs_f['B'][gt_airice[0]-1]=np.log(1)
    initial_probs_f['nB'][gt_airice[0]-1]=np.log(1e-20)

    airice_feedback = viterbi('airice',0,'na',obs_p_b_f, obs_p_notb_f, initial_probs_f, transition_probs)
    airice_feedback = np.roll(airice_feedback,gt_airice[1]-1)

    # ice rock layers
    icerock_simple = []
    for n in range(image_array.shape[1]):
            compare_p = np.subtract(obs_p_b[:,n][int(airice_simple[n])+30:],obs_p_notb[:,n][int(airice_simple[n])+30:])
            icerock_simple = np.append(icerock_simple, argmax(compare_p)+30+airice_simple[n])

    icerock_hmm = viterbi('icerock', 30, airice_hmm, obs_p_b, obs_p_notb, initial_probs, transition_probs)

    obs_p_b_f= np.roll(obs_p_b, -(gt_icerock[1]-1), axis =1)
    obs_p_notb_f = np.roll(obs_p_notb, -(gt_icerock[1]-1), axis =1)
    obs_p_b_f[gt_icerock[0], gt_icerock[1]]=np.log(1)
    obs_p_notb_f[gt_icerock[0], gt_icerock[1]]=np.log(1e-20)


    initial_p_f = [np.repeat(np.log(.5), 175),np.repeat(np.log(.5), 175)]
    initial_probs_f = dict(zip(hidden_states, initial_p_f))
    initial_probs_f['B'][gt_icerock[1]-1]=np.log(1)
    initial_probs_f['nB'][gt_icerock[1]-1]=np.log(1e-20)


    icerock_feedback = viterbi('icerock', 30, airice_feedback, obs_p_b_f, obs_p_notb_f, initial_probs, transition_probs)
    icerock_feedback = np.roll(icerock_feedback,gt_icerock[1]-1)



    # Now write out the results as images and a text file
    write_output_image("air_ice_output.png", input_image, airice_simple, airice_hmm, airice_feedback, gt_airice)
    write_output_image("ice_rock_output.png", input_image, icerock_simple, icerock_hmm, icerock_feedback, gt_icerock)
    with open("layers_output.txt", "w") as fp:
        for i in (airice_simple, airice_hmm, airice_feedback, icerock_simple, icerock_hmm, icerock_feedback):
            fp.write(str(i) + "\n")
