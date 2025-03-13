import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

def network_colormap(nx_graph, pos, y_list, node_list, attr_list, name='Ground Truth'):
    label_0 = [node_list[x] for x in range(len(node_list)) if y_list[x] == 0]
    label_1 = [node_list[x] for x in range(len(node_list)) if y_list[x] == 1]
    label_2 = [node_list[x] for x in range(len(node_list)) if y_list[x] == 2]
    options = {
        "node_size": 100,
        "edgecolors": "black"
    }
    plt.title(name)
    nx.draw_networkx_edges(nx_graph, pos, alpha=attr_list)
    nx.draw_networkx_nodes(nx_graph, pos=pos, nodelist=label_0, node_color='grey', label='Not interface', **options)
    nx.draw_networkx_nodes(nx_graph, pos=pos, nodelist=label_2, node_color='skyblue', label='Ellipro', **options)
    nx.draw_networkx_nodes(nx_graph, pos=pos, nodelist=label_1, node_color='lightgreen', label='CIPS', **options)
    plt.axis('off')
    plt.legend()

def plot_network(node_list, pred_list, true_list, edge, attr, pdbId, fold_folder, networkx_seed):
    
    # Create edge_index
    edge_df = pd.DataFrame(edge.detach().cpu().numpy().T,columns=['source','target'])
    edge_df = edge_df[edge_df.source < edge_df.target]
    
    edge_df.loc[:,'source'] = edge_df.source.map(lambda x: node_list[x])
    edge_df.loc[:,'target'] = edge_df.target.map(lambda x: node_list[x])
    # Create edge_attribute_list
    attr_value = ((attr - attr.min())/(attr.max() - attr.min()) *0.7).detach().cpu().numpy()

    # Draw
    G = nx.from_pandas_edgelist(edge_df)
    plt.figure(figsize = (20,10))
    pos = nx.spring_layout(G, seed=networkx_seed)
    plt.subplot(121)
    network_colormap(G, pos, true_list, node_list, attr_value, 'Ground Truth')
    plt.subplot(122)
    network_colormap(G, pos, pred_list, node_list, attr_value, 'Prediction')
    plt.savefig(os.path.join(fold_folder, f'{pdbId}.png'), bbox_inches='tight')
    plt.close('all')
