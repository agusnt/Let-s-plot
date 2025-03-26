#!/usr/bin/env python3
'''
Automatic generation of figures from a json file

@Author: Navarro-Torres, Agustin 
@Email: agusnt@unizar.es, agusnavarro11@gmail.com

Parameters:
    1 : json file
    2 : Output file
'''

import math 
import matplotlib

import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import json, sys
import argparse

###############################################################################
# Auxiliary functions
###############################################################################
def eprint(*args, **kwargs): print(*args, file=sys.stderr, **kwargs)

###############################################################################
# JSON functions
###############################################################################
def read_json(fname):
    '''
    Read json file
    
    Parameters:
        fname : file to read

    Return: file as dictionary
    '''
    f = open(fname, 'r')
    foo = json.loads(f.read())
    f.close()
    return  foo

###############################################################################
# Graph Aux functions
###############################################################################
def data_to_array(fname):
    '''
    Convert the json with the data to three arrays: x, y, and values

    Parameter:
        fname : json name file with the data 

    Return:
        x : array with x values
        y : array with y values
        v : array with values
        c : array with colors 
        m : array with marks
        l : array for use as legend
        a : array with extra arguments
    '''
    x = []
    y = []
    v = []
    c = []
    m = []
    mc = []
    l = []
    a = []

    data = read_json(fname)
    
    for i in data:
        if 'x' in i: x.append(i['x'])
        if 'y' in i: y.append(i['y'])
        if 'value' in i: v.append(i['value'])
        if 'color' in i: c.append(i['color'])
        if 'marker' in i: m.append(i['marker'])
        if 'edge_color' in i: mc.append(i['edge_color'])
        if 'legend' in i: l.append(i['legend'])
        else: l.append(True)
        if 'args' in i: a.append(i['args'])
        else: a.append({})

    return x, y, v, c, m, mc, l, a

def array_to_np(x, y, v, s):
    '''
    Convert three arrays (x, y and values) to a numpy matrix

    Parameters:
        x : array with the x index 
        y : array with the y index 
        v : array with the values
        s : size of matrix

    Return: a numpy matrix
    '''
    matrix = np.zeros(s)
    for i, j, k in zip(x, y, v): matrix[i][j] = k
    return matrix

def set_annotate(ax, args, fig):
    '''
    Set values to print an annotation in the graph

    Parameters:
        ax : axis graph
        args : arguments arguments
        fig : figure object
    '''
    arg = args['args'] if 'args' in args else {}
    annotate(ax, args['str'], args['x'], args['y'], 0, 0, arg, fig)

def annotate(ax, v, x, y, xd, yd, arg, fig):
    '''
    Print an annotation in the graph

    Parameters:
        ax : axis graph
        v : value to write
        x : initial x position
        y : initial y position
        xd : extra end x position
        yd : extra end y position
        arg : extra arguments
        fig : figure object
    '''
    # Simulate a C/C++ static var for saving the previous annotation
    if not hasattr(annotate, 'prev'): annotate.prev = []

    def intersects(first, second):
        # Check collistion of the BBox

        a = first[0].get_window_extent().transformed(ax.transData.inverted())
        b = second[0].get_window_extent().transformed(ax.transData.inverted())

        for i, pad in [(a, first[1]), (b, second[1])]:
            i.x0 -= pad
            i.y0 += pad
            i.x1 += pad
            i.y1 -= pad
        if (b.x0 >= a.x0 and b.x0 <= a.x1 and b.y1 <= a.y0 and b.y1 >= a.y1) or\
            (b.x1 >= a.x0 and b.x1 <= a.x1 and b.y0 >= a.y1 and b.y0 <= a.y0):
            return True
        return False

    is_right = False
    # How to move the annotation (x, y)
    clock_position_dx = 0
    clock_position = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    while not is_right:
        is_right = True # Should work

        # Print the annotation
        if 'label' in arg: v_ = '{} {}'.format(arg['label'], v)
        else: v_ = v

        ann = ax.annotate(v_, [x, y], [x+xd, y+yd], annotation_clip=False, **arg)

        # Get the position of the axis
        fig.draw_without_rendering() # Draw it to force it to get the right one

        # Get the extra path of the BBox 
    #    pad = ann.get_bbox_patch().get_boxstyle().pad

    #    for i in annotate.prev:
    #        if intersects((ann, pad), i): 
    #            is_right = False
    #            ann.remove()
    #            break
    #    clock = clock_position[clock_position_dx]
    #    if clock[0] != 0:
    #        if xd < 0: xd *= -1
    #        xd += 0.1
    #        xd *= clock[0]
    #    if clock[1] != 0:
    #        if yd < 0: yd *= -1
    #        yd += 0.1
    #        yd *= clock[1]
    #    clock_position_dx = (clock_position_dx + 1) % len(clock_position)
    #annotate.prev.append((ann, pad))

###############################################################################
# Graph Plot functions
###############################################################################
def set_fig(info = {}, rows=1, columns=1, size=None):
    if size: return plt.subplots(nrows=rows, ncols=columns, figsize=size, **info)
    else: return plt.subplots(rows, columns, **args)

def subplot_adjust(info):
    plt.subplots_adjust(**info)

def get_ax(dx, dy, axs):
    '''
    Return the sub element from the axis

    TODO: Move from try and catch code to a more sophisticated one

    Parameter:
        dx : axis x
        dy : axis y
        axs : array of figures

    Return: the ax
    '''
    try:
        if len(axs.shape) == 2: return axs[dx][dy] # Is a matrix figure
        elif len(axs.shape) == 1: return axs[dx] # Is a line figure
    except: return axs # Is an standalone figure

def set_axis(ax, axis, info):
    '''
    Format axis

    Parameters:
        ax: axis 
        axis : 'x', 'y' or 'common'
        info : format of the axis
    '''
    args_labels = info['args_labels'] if 'args_labels' in info else {}
    args_txt_label = info['args_txt_label'] if 'args_txt_label' in info else {}

    # Set visibility of the axis, not axis needed
    if 'visibility' in info and axis == 'common':
        ax.spines[info['visibility']['spine']].set_visible(info['visibility']['value'])

    # Set a fancy cut
    if 'fancy_cut' in info and axis == 'common':
        d = info['fancy_cut']['size']
        s = info['fancy_cut']['sloppy']
        if info['fancy_cut']['pos'] == 'left':
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
            ax.plot((1-d/s, 1+d/s), (-d, +d), **kwargs)
            ax.plot((1-d/s, 1+d/s), (1-d, 1+d), **kwargs)
        elif info['fancy_cut']['pos'] == 'right':
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
            ax.plot((-d, +d), (1-d/s, 1+d/s), **kwargs)
            ax.plot((-d, +d), (-d/s, +d/s), **kwargs)
        elif info['fancy_cut']['pos'] == 'bottom':
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
            ax.plot((-d, +d), (-d/s, d/s), **kwargs)
            ax.plot((1-d, 1+d), (-d/s, +d/s), **kwargs)
        elif info['fancy_cut']['pos'] == 'top':
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
            ax.plot((-d, +d), (1-d/s, 1+d/s), **kwargs)
            ax.plot((1-d, 1+d), (1-d/s, 1+d/s), **kwargs)

    if 'scale' in info:
        if axis == 'x': ax.set_xscale(info['scale'])
        elif axis == 'y': ax.set_yscale(info['scale'])

    if 'format' in info and info['format'] == 'scalar':
        if axis == 'x': ax.xaxis.set_major_formatter(ScalarFormatter())
        elif axis == 'y': ax.yaxis.set_major_formatter(ScalarFormatter())

    if 'params' in info:
        if axis == 'x': ax.tick_params(axis='x', **info['params'])
        elif axis == 'y': ax.tick_params(axis='y', **info['params'])

    if 'label' in info: # Label
        if axis == 'x': ax.set_xlabel(info['label'])
        elif axis == 'y': ax.set_ylabel(info['label'], **args_txt_label)

    if 'ticks' in info: # Ticks
        if axis == 'x': ax.set_xticks(info['ticks'])
        elif axis == 'y': ax.set_yticks(info['ticks'])

    if 'ticks_label' in info: # Labels of ticks
        if axis == 'x': ax.set_xticklabels(info['ticks_label'], **args_labels)
        elif axis == 'y': ax.set_yticklabels(info['ticks_label'], **args_labels)

    if 'minor_ticks' in info: # Minor ticks
        if axis == 'x': ax.set_xticks(info['minor_ticks'], minor=True, **args_labels)
        elif axis == 'y': ax.set_yticks(info['minor_ticks'], minor=True, **args_labels)

    if 'max' in info and 'min' in info: # Limits
        if axis == 'x': ax.set_xlim((info['min'], info['max']))
        elif axis == 'y': ax.set_ylim((info['min'], info['max']))

    if 'grid' in info and info['grid'] == True: # Grid 
        if axis == 'x': ax.grid(axis='x')
        elif axis == 'y': ax.grid(axis='y')

    if 'subgrid' in info and info['subgrid'] == True: # Sub Grid 
        if axis == 'x': ax.grid(axis='x', which='minor', linestyle='--')
        elif axis == 'y': ax.grid(axis='y', which='minor', linestyle='--')

    if 'margin' in info: # Margin
        if axis == 'x': ax.margins(x=info['margin'])
        elif axis == 'y': ax.margins(y=info['margin'])

    if 'tick_params' in info:
        if axis == 'x': ax.tick_params(axis='x', **info['tick_params'])
        elif axis == 'y': ax.tick_params(axis='y', **info['tick_params'])

def graph_format(ax, info):
    '''
    Set specific design for the graph
    
    Parameters:
        ax : graph to set format
        info : design info
    '''

    if 'axis' in info: # Axis format}}
        if 'x' in info['axis']: set_axis(ax, 'x', info['axis']['x'])
        if 'y' in info['axis']: set_axis(ax, 'y', info['axis']['y'])
        if 'common' in info['axis']: set_axis(ax, 'common', info['axis']['common'])

    if 'title' in info: ax.set_title(info['title']) # Title format

    if 'legend' in info:
        legends = []
        # Set legend
        for idx, i in enumerate(info['legend']):
            h, l = ax.get_legend_handles_labels()
            legends_elements = []
            for ii in i['elems']:
                # Identify if there is a fig_placeholder
                if ii == 'mt_lg_placeholder': 
                    legends_elements.append(
                        mpatches.Patch(facecolor='white', edgecolor='white')
                    )
                    continue

                if ii not in l and ii != 'mt_lg_placeholder': 
                    # Show an error because there is something miss in legend
                    if 'extra' not in info['legend'][idx]:
                        eprint("{} {} {} {}".format(
                            "I can't not found", ii, "in the data, but I can't",
                            "either found an 'extra' field with its information"
                        ))
                        continue

                    legends_elements.append(
                        mpatches.Patch(
                            **info['legend'][idx]['extra'][ii], 
                            label=ii
                        )
                    )
                    continue # This legend is not in our graph
                legends_elements.append(h[l.index(ii)])
            args = i['args'] if 'args' in i else {}
            # ax.legend(handles=legends_elements, **args)
            if idx != len(info['legend']) - 1:
                legends.append(ax.legend(handles=legends_elements, **args))
            else:
                for ii in legends: ax.add_artist(ii)
                ax.legend(handles=legends_elements, **args)

    if 'hline' in info: 
        for i in info['hline']: ax.axhline(**i)

###############################################################################
# Graph functions
###############################################################################
def fig_heatmap(ax, jgraph):
    '''
    Do a heatmap graph

    Parameters:
        fig : figure object
        axs : axis object
        jgraph : json with all the information for this graph
    '''
    # Get the data
    x, y, v, _, _, _, _, _= data_to_array(jgraph['data'])
    matrix = array_to_np(y, x, v, (jgraph['size_y'], jgraph['size_x']))
    # Remove 0
    matrix[matrix == 0] = np.nan

    # CMP
    cmap = cm.gray_r
    if jgraph['cmap'] == 'gray': cmap = cm.gray_r
   
    #Args
    args = jgraph['args'] if 'args' in jgraph else {}

    # Plot the figure
    sns.heatmap(matrix, cmap=cmap, ax=ax, **args)

def fig_plot(ax, jgraph):
    '''
    Do a plot graph

    Parameters:
        fig : figure object
        ax : axis object
        jgraph : json with all the information for this graph
    '''
    # Get the data
    x, y, v, c, m, _, ll, argsData = data_to_array(jgraph['data'])
    
    # X and Y must be equal
    assert len(x) == len(y), "X and Y has different size"

    # Set all array same size
    def gen_array(a, b): return [b for _ in range(0, len(a))] # Gen a new array
    if len(x) != len(c): c = gen_array(x, 'black')
    if len(x) != len(c): c = gen_array(x, 'o')

    #Args
    args = jgraph['args'] if 'args' in jgraph else {}
    
    # Split every line to print into different lines
    l = {}
    for i, j, k, n, o, a in zip(x, y, c, m, v, argsData):
        if o not in l: l[o] = {'x': [], 'y': [], 'c': [], 'm': [], 'v': [], 'a': {}}
        l[o]['x'].append(i)
        l[o]['y'].append(j)
        l[o]['c'].append(k)
        l[o]['m'].append(n)
        l[o]['v'].append(o)
        l[o]['a'] = a

    # Plot it
    for i in l:
        if all(ll):
            ax.plot(l[i]['x'], l[i]['y'], c=l[i]['c'][0], marker=l[i]['m'][0], **args, **l[i]['a'], label=i)
        else:
            ax.plot(l[i]['x'], l[i]['y'], c=l[i]['c'][0], marker=l[i]['m'][0], **args, **l[i]['a'])

def fig_scatter(ax, jgraph):
    '''
    Do a scatter graph

    Parameters:
        fig : figure object
        ax : axis object
        jgraph : json with all the information for this graph
    '''
    # Get the data
    x, y, v, c, m, e, l, _ = data_to_array(jgraph['data'])
    
    # X and Y must be equal
    assert len(x) == len(y), "X and Y has different size"

    # Set all array same size
    def gen_array(a, b): return [b for _ in range(0, len(a))] # Gen a new array
    if len(x) != len(c): c = gen_array(x, 'black')
    if len(x) != len(c): c = gen_array(x, 'o')

    #Args
    args = jgraph['args'] if 'args' in jgraph else {}

    # Plot the figure
    for i, j, k, o, n, q, ed in zip(x, y, c, m, v, l, e):
        # This iteration is not the more optimal thing, but is easier for 
        # programming, since I can change the color/marker of each point
        # without making exception. TLDR: not optimal bc I made everything an 
        # exception.
        if q: ax.scatter(i, j, c=k, edgecolors=ed, marker=o, label=n, **args) # With label
        else: ax.scatter(i, j, c=k, edgecolors=ed, marker=o, **args) # Without label

def fig_bar(ax, jgraph, fig):
    '''
    Do a bar graph

    Parameters:
        fig : figure object
        ax : axis object
        jgraph : json with all the information for this graph
    '''

    def get_bar_pos(jgraph, size_bar=0.8):
        '''
        Set the bar positions

        Parameters:
            jgraph : graph options

        Return:
            positions of bars, and width of bars
        '''
        lg_placeholder = None
        if 'mt_lg_placeholder' in jgraph['order']: 
            # Remove legend placeholder
            lg_placeholder = jgraph['order'].index('mt_lg_placeholder')
            jgraph['order'].remove('mt_lg_placeholder')

        order = jgraph['order'] if 'order' in jgraph else set(v)
        pos = []

        # Auxiliary functions
        # Get initial position
        def get_init(length, size): return size if (length % 2) != 0 else size / 2
        def set_pos(length, init, func): 
            # Set positions
            for _ in range(0, length//2):
                pos.append(init)
                init = func(init, size)

        size = size_bar / len(order) # size of bars

        # Left position
        set_pos(len(order), get_init(len(order), size) * -1, lambda a, b: a-b)
        # Medium
        if len(order) % 2 != 0: pos.append(0) 
        # Right positions
        set_pos(len(order), get_init(len(order), size), lambda a, b: a+b)

        # Related label and position
        loc = {}
        for i, j in zip(order, sorted(pos)): loc[i] = j

        # Set again lg_placeholder
        if lg_placeholder is not None:
            jgraph['order'].insert(lg_placeholder, 'mt_lg_placeholder')

        return loc, size
    # Get the data
    x, y, v, c, m, mc, l, _ = data_to_array(jgraph['data'])

    # If y value is this is an stacked bar
    if type(y) == list: 
        y = [i for i in y] # Enable stacked bar, y
        c = [i for i in c] # Enable stacked bar, color
        m = [i for i in m] # Enable stacked bar, pattern
        mc = [i for i in mc] # Enable stacked bar, pattern color

    # Get bar width
    if 'size_bar' in jgraph:
        loc, size = get_bar_pos(jgraph, size_bar = jgraph['size_bar'])
    else: loc, size = get_bar_pos(jgraph)

    args = jgraph['args'] if 'args' in jgraph else {}

    # Plot the figure
    for dx, (i, j, k, n, q) in enumerate(zip(x, y, v, c, l)):
        if k not in loc: continue # Some bars maybe are not interesting

        if type(j) == list: 
            bar = [0] + j[:-1]
            b = [sum(bar[:ii+1]) for ii in range(0, len(bar))]
            maxj = max(j)
        else: 
            maxj = j
            b = 0

        ec = [] 
        if len(mc) == 0:
            if type(j) == list: ec = ['black' for _ in j]
            else: ec = 'black'
        else:
            if type(mc[dx]) == list and len(mc[dx]) == 0: ec = ['black' for _ in y]
            elif mc[dx] == '': ec = 'black'
            else: 
                ec = mc[dx]

        # With and without label
        # Bottom should go to another type of bar, stacked
        #if q: ax.bar(i + loc[k], j, color=n, width=size, hatch=m[dx], edgecolor=ec, bottom=b, **args, label=k)
        #else: ax.bar(i + loc[k], j, color=n, width=size, hatch=m[dx], edgecolor=ec, bottom=b, **args)

        if 'bottom' in args: 
            j -= args['bottom']

        if q: ax.bar(i + loc[k], j, color=n, width=size, hatch=m[dx], edgecolor=ec, **args, label=k)
        else: ax.bar(i + loc[k], j, color=n, width=size, hatch=m[dx], edgecolor=ec, **args)
        # Ensure that the border is always black
        ax.bar(i + loc[k], j, color='#00000000', width=size, edgecolor='black', **args)

        # If stacked bar set border around it
        #if 'EXT_BORDER' in ['args']:
        if type(j) == list and len(j) > 1:
            ax.bar(i + loc[k], sum(j), color='none', width=size, edgecolor='black', zorder=99)


        if 'annotate' in jgraph and k in jgraph['annotate']:
            decimals = jgraph['annotate'][k]['round']
            maxxj = math.floor(maxj*10**decimals) / (10**decimals)
            minnj = math.floor(maxj*10**decimals) / (10**decimals)
            if maxxj > jgraph['axis']['y']['max'] or \
                    minnj <= jgraph['axis']['y']['min'] or \
                    ('always' in jgraph['annotate'][k] and \
                     jgraph['annotate'][k]['always']):

                # Annotate for this bar if the value is bigger than x
                dic_annotate = jgraph['annotate'][k]

                # Get extra info
                r = dic_annotate['round'] if 'round' in dic_annotate else 1
                r = dic_annotate['decimals_show'] if 'decimals_show' in dic_annotate else r
                arg = dic_annotate['args'] if 'args' in dic_annotate else {}

                # Relative position
                xd = dic_annotate['x'] if 'x' in dic_annotate else 0
                yd = dic_annotate['y'] if 'y' in dic_annotate else 0

                # Position x and y, and value to write 
                x = i + loc[k]
                y = jgraph['axis']['y']['max']

                # If we always write the annotations we have to change the y
                if maxxj < jgraph['axis']['y']['max']: y = maxj

                if (r != 0): v = "{}".format(round(maxj, r))
                else: v = "{}".format(int(maxj))

                annotate(ax, v, x, y, xd, yd, arg, fig)
    
    
def main(chart : str, output : str, inv = False):
    jgraph = read_json(chart) # Read json config file

    # Subplot size
    rows = 1 if 'rows' not in jgraph else jgraph['rows']
    columns = 1 if 'columns' not in jgraph else jgraph['columns']
    figsize = None if 'figsize' not in jgraph else \
            (int(jgraph['figsize'].split(',')[0]), \
            int(jgraph['figsize'].split(',')[1]))

    # Arguments of the plot
    if 'args' in jgraph: 
        for i in jgraph['args']: matplotlib.rc(i, **jgraph['args'][i])
    # Create subplot
    if 'splt_args' in jgraph:
        fig, axs = set_fig(jgraph['splt_args'], rows=rows, columns=columns, size=figsize)
    else:
        fig, axs = set_fig(rows=rows, columns=columns, size=figsize)
    # Adjust subplots
    if 'splt_adjust' in jgraph:
        subplot_adjust(jgraph['splt_adjust'])

    #General Annotion
    if 'annotate' in jgraph:
        for ii in jgraph['annotate']: 
            if 'args' in ii: fig.text(ii['x'], ii['y'], ii['txt'], **ii['args'])
            else: fig.text(ii['x'], ii['y'], ii['txt'])

    # Iterate over all do graphs
    for i in jgraph['graphs']:
        if rows > 1 and columns == 1: ax = get_ax(i['dy'], i['dx'], axs) # Get position of the graph
        else: ax = get_ax(i['dx'], i['dy'], axs) # Get position of the graph

        if i['type'] == 'heatmap': fig_heatmap(ax, i)
        elif i['type'] == 'scatter': fig_scatter(ax, i)
        elif i['type'] == 'bar': fig_bar(ax, i, fig)
        elif i['type'] == 'plot': fig_plot(ax, i)

        if 'annotate' in i and 'general' in i['annotate']:
            for ii in i['annotate']['general']: set_annotate(ax, ii, fig)

        graph_format(ax, i)

        if inv:
            # Only show the legend
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.grid(False)
            ax.grid(False, which='minor')
            ax.set_yticks([], minor=True)
            ax.set_frame_on(False)  # Oculta el borde del gráfico
            for line in ax.lines:
                line.set_visible(False)

            for patch in ax.patches:
                patch.set_visible(False)

            for collection in ax.collections:
                collection.set_visible(False)

    graph_format(fig, jgraph)

    # Save fig as pdf
    fig.savefig(output, bbox_inches='tight')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate figure from json file')
    parser.add_argument('chart', type=str, help='Json file with the chart')
    parser.add_argument('output', type=str, help='Output file')
    parser.add_argument('-i', action='store_true', help='Only print legend')
    args = parser.parse_args()
    if args.i:
        main(args.chart, args.output, inv=True)
    else:
        main(args.chart, args.output)

    
