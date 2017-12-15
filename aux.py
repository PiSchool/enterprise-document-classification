# -*- coding: utf-8 -*-
"""
Auxiliary functions: plotting, creating pdfs etc

@author: robi
"""
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
import numpy as np

def average_hist(dataset, nitems, C, dataset_path):
    """
    Calculate an online version of the average histogram + histogram variance over each class
    """
    import cv2
    import skimage.io as skio
    # Average histogram for each class
    hist_set={}
    for c in dataset['train']:
        if c not in hist_set:
            hist_set[c]= np.zeros((256,1))
        for ii in range(int( nitems[0,C[c]] )):
            hist_set[c] += cv2.calcHist([skio.imread(dataset_path+'images/'+dataset['train'][c].values[ii,0])],[0],None,[256],[0,256])
            print(ii)
        hist_set[c] = hist_set[c]/nitems[0,C[c]]
        
    return hist_set

def plot_sg_hashm(dimgh, rot, KK):
    # Show a Tabel of SingleClass Hash Distance Matrix
    f, axarr = plt.subplots(4, 4)
    f.set_size_inches(8,6)
    f.suptitle("SingleClass Phash Distance Matrix - PureRotation (-60,+50)",
               fontweight=4)
    for ii in range(4):
        for jj in range(4):
            cax= axarr[ii,jj].imshow(dimgh[(ii*4+jj)*rot.size:(ii*4+jj+1)*rot.size,
                                (ii*4+jj)*rot.size:(ii*4+jj+1)*rot.size], cmap='hot')
            plt.colorbar(cax)
            axarr[ii,jj].set_title(KK[ii*4+jj], fontsize=8)
            axarr[ii,jj].set_yticks( range(0,rot.size,1) )        
            axarr[ii,jj].set_yticklabels( np.round(rot, decimals=2), rotation=0, fontsize=5 )
            if ii==3:
                axarr[ii,jj].set_xticks( range(0,rot.size,1) )
                axarr[ii,jj].set_xticklabels( np.round(rot, decimals=2), rotation=90, fontsize=5 )
            else:
                plt.setp([a.get_xticklabels() for a in axarr[ii, :]], visible=False)    
    
    return f

def plot_mc_hashm(dimgh, rot, KK):
    """
    Plot the image of the multiclass phash distance matrix
    """
    # Show the Multiclass Hash Distance Matrix
    fig= plt.figure()
    plt.title("Single-MultiClass Phash Distance Matrix - PureRotation (-60,+50)",
              loc='center',fontweight=4)
    plt.imshow(dimgh, cmap='hot')
    plt.colorbar(), plt.grid('off')
    plt.xticks( range( 0,rot.size*len(KK),rot.size), KK, rotation=90 )
    plt.yticks( range( 0,rot.size*len(KK),rot.size), KK, rotation=0 )
    
    return fig

def hclass(nitems, K):
    # Show histogram  of the number of samples for each class in the training set
    # Inputs:
    # nitems= numpy vector of elements for each class in the training set
    # K= number of classes
    # 
    # Outputs:
    # None
    plt.bar( np.linspace(0,K-1, num=K), nitems )
    plt.show()

def multi_plot(images, titles):
    # Multi-image plotting function
    
    r= images.shape[0]
    c= images.shape[1]
    
    plt.figure(1, figsize=(8, 6), dpi=128, facecolor='w', edgecolor='k')
    for i in range(r):
        for jj in range(c):
            plt.subplot(r,c,i*r+jj),plt.imshow(images[i*r],'gray')
            plt.title(titles[i*r]), plt.xticks([]), plt.yticks([])
        

def createpdf(inputs, outfile, results_path, **kwargs):
    # This function expects:
    # inputs: dictionary of image groups, where for each keyword (corresponding
    #           to a class) we have a dictionary: (images,titles)
    # outfile: name of the output file
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 8}
    
    matplotlib.rc('font', **font)
    
    
    ti= datetime.datetime.now().strftime("%I:%M%p_%B%d%Y")
    pp = PdfPages(results_path+outfile+'_'+ti+'.pdf')
    
    for ii in inputs:
        
        # Save to pdf page
        pp.savefig(ii)
    
    #Close the pdf
    pp.close()