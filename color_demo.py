#!/bin/python3
'''
Color demo

@Author: Navarro-Torres, Agustin 
@Email: agustin.navarro@um.es, agusnavarro11@gmail.com
'''

import matplotlib.pyplot as plt


def fn(scale, name):
    x = [idx+0.5 for idx, _ in enumerate(scale)]
    y = [10 for _ in scale]
    
    _, ax = plt.subplots()
    
    ax.bar(x, y, color=scale, width=1, edgecolor='black')
    
    ax.tick_params(left=False)
    
    ax.set_xticks(x)
    ax.set_yticks(y)
    
    ax.set_xticklabels(scale, rotation=45)
    ax.set_yticklabels([])
    
    ax.set_ylim([0, 10])
    ax.set_xlim([0, len(x)])
    
    plt.savefig(name, bbox_inches='tight')

# Gray scale
fn(['#000000', '#282828', '#474747', '#646464', '#7e7e7e', '#9b9b9b', '#bdbdbd', 
    '#e7e7e7', '#ffffff'], 'bw.pdf')
# IBM colors (BW safe)
fn(['#648fff', '#785ef0', '#dc267f', '#fe6100', '#ffb000', '#000000', '#ffffff'], 
   'ibm.pdf')
# Tol (BW safe)
fn(['#ffffff', '#ddaa33', '#bb5566', '#004488', '#000000'], 'tol.pdf')
# Okabe Ito
fn(['#ffffff', '#E69F00', '#56B4E9', '#009E73', '#F0E442', 
    '#0072B2', '#D55E00', '#CC79A7', '#000000'], 'okabe_ito.pdf')
# Big Achromatopsia
fn(['#ffffff', '#E8F086', '#6FDE6E', '#A691AE', '#FF4242', 
    '#235FA4', '#104575', '#000000'], 'colorful.pdf')
