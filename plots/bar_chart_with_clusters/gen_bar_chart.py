import json
import argparse
import tempfile
import sys
from pprint import pprint as pp
from statistics import geometric_mean

sys.path.append('../..')
import gen_fig

PALETTES = {
    'gray_scale': ['#000000', '#282828', '#474747', '#646464', '#7e7e7e', '#9b9b9b', '#bdbdbd', '#e7e7e7', '#ffffff'],
    'okabe_ito': ['#ffffff', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000'],
    'big_achromatopsia': ['#ffffff', '#E8F086', '#6FDE6E', '#A691AE', '#FF4242', '#235FA4', '#104575', '#000000'],
    
    # BW safe palettes
    "ibm": ['#648fff', '#785ef0', '#dc267f', '#fe6100', '#ffb000', '#000000', '#ffffff'],
    'tol': ['#ffffff', '#ddaa33', '#bb5566', '#004488', '#000000'], 
}


def gen_bar_data(data : json, args : argparse.Namespace):
    # check that all clusters contains the same number of series
    assert all(len(x['data']) == len(data['series']) for x in data['clusters'])
    
    # check that we have enough colors
    assert len(PALETTES[args.palette]) >= len(data['series'])
    
    # assign an index to each cluster and serie
    cluster_idx = {x['name']: idx for idx, x in enumerate(data['clusters'])}
    series_idx = {x: idx for idx, x in enumerate(data['series'])}
    
    bar_data = []
    for cluster in data['clusters']:
        for serie in data['series']:
            bar_data.append({
                'x': cluster_idx[cluster['name']],
                'y': cluster['data'][series_idx[serie]],
                'value': serie,
                'color': PALETTES[args.palette][series_idx[serie]],
                'edge_color': 'black',
                'marker': '',
                'legend': True
            })
            
    if args.geomean:
        print('--Adding geomean cluster')
        for serie in data['series']:
            bar_data.append({
                'x': len(cluster_idx),
                'y': geometric_mean([x['data'][series_idx[serie]] for x in data['clusters']]),
                'value': serie,
                'color': PALETTES[args.palette][series_idx[serie]],
                'edge_color': 'black',
                'marker': '',
                'legend': True
            })

    return bar_data


def gen_bar_chart(args : argparse.Namespace):
    with open(args.data, 'r') as f:
        data = json.load(f)
        
    with open(args.chart_template, 'r') as f:
        chart_template = json.load(f)
    
    formatted_data = gen_bar_data(data, args)
    
    # Dump the data into a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as data_file:
        json.dump(formatted_data, data_file)

    # Fill the chart template
    cluster_names = [x['name'] for x in data['clusters']] + (['Geomean'] if args.geomean else [])
    series_names = data['series']
    
    chart_template['graphs'][0]['data'] = data_file.name
    chart_template['graphs'][0]['axis']['x']['ticks'] = [x for x in range(len(cluster_names))]
    chart_template['graphs'][0]['axis']['x']['ticks_label'] = cluster_names
    chart_template['graphs'][0]['axis']['x']['min'] = -0.5
    chart_template['graphs'][0]['axis']['x']['max'] = len(cluster_names) - 0.5
    chart_template['graphs'][0]['legend'][0]['elems'] = series_names
    chart_template['graphs'][0]['order'] = series_names

    # Dump the chart template into a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as chart_file:
        json.dump(chart_template, chart_file)
        
    # Generate the figure
    gen_fig.main(chart_file.name, args.output)
    print(f'Figure saved in {args.output}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate bar chart')
    parser.add_argument('data', type=str, help='File with the data')
    parser.add_argument('chart_template', type=str, help='Chart template')
    parser.add_argument('output', type=str, help='Output file')
    parser.add_argument('--palette', choices=PALETTES.keys(), type=str, default='ibm', help='Color palette')
    parser.add_argument('--geomean', action='store_true', help='Add a new cluster with the geometric mean of the series')
    args = parser.parse_args()
    
    gen_bar_chart(args)



