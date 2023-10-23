import os
from matplotlib import tri as mtri
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import copy
from torch import Tensor
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
import networkx as nx
from networkx.algorithms import community

def save_plots(args, losses, test_losses, velo_val_losses):
    model_name='model_nl'+str(args.num_layers)+'_bs'+str(args.batch_size) + \
               '_hd'+str(args.hidden_dim)+'_ep'+str(args.epochs)+'_wd'+str(args.weight_decay) + \
               '_lr'+str(args.lr)+'_shuff_'+str(args.shuffle)+'_tr'+str(args.train_size)+'_te'+str(args.test_size)

    if not os.path.isdir(args.postprocess_dir):
        os.mkdir(args.postprocess_dir)

    PATH = os.path.join(args.postprocess_dir, model_name + '.pdf')

    f = plt.figure()
    plt.title('Losses Plot')
    plt.plot(losses, label="training loss" + " - " + args.model_type)
    plt.plot(test_losses, label="test loss" + " - " + args.model_type)
    #if (args.save_velo_val):
    #    plt.plot(velo_val_losses, label="velocity loss" + " - " + args.model_type)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.show()
    f.savefig(PATH, bbox_inches='tight')




def make_animation(gs, path, name , skip = 2, save_anim = True, plot_variables = False):
    '''
    input gs is a dataloader and each entry contains attributes of many timesteps.

    '''
    print('Generating velocity fields...')
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    num_steps = len(gs) # for a single trajectory
    num_frames = num_steps // skip
    print(num_steps)
    def animate(num):
        step = (num*skip) % num_steps
        traj = 0

        bb_min = gs[0].x[:, 0:2].min() # first two columns are velocity
        bb_max = gs[0].x[:, 0:2].max() # use max and min velocity of gs dataset at the first step for both 
                                          # gs and prediction plots
        count = 0


        ax.cla()
        ax.set_aspect('equal')
        ax.set_axis_off()
        
        pos = gs[step].mesh_pos 
        faces = gs[step].cells
        velocity = gs[step].x[:, 0:2]
        title = 'Ground truth:'
        

        triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
        mesh_plot = ax.tripcolor(triang, velocity[:, 0], vmin= bb_min, vmax=bb_max,  shading='flat' ) # x-velocity
        ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)


        ax.set_title('{} Trajectory {} Step {}'.format(title, traj, step), fontsize = '20')
        #ax.color

        #if (count == 0):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        clb = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
        clb.ax.tick_params(labelsize=20) 
        
        clb.ax.set_title('x velocity (m/s)',
                            fontdict = {'fontsize': 20})
        return fig,

    # Save animation for visualization
    if not os.path.exists(path):
        os.makedirs(path)
    
    if (save_anim):
        gs_anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=1000)
        writergif = animation.PillowWriter(fps=10) 
        anim_path = os.path.join(path, '{}_anim.gif'.format(name))
        gs_anim.save( anim_path, writer=writergif)
        plt.show(block=True)
    else:
        pass

def visualize(loader, best_model, file_dir, args, gif_name, stats_list,
              delta_t = 0.01, skip = 1):

    best_model.eval()
    device = args.device
    viz_data = {}
    gs_data = {}
    eval_data = {}
    viz_data_loader = copy.deepcopy(loader)
    gs_data_loader = copy.deepcopy(loader)
    eval_data_loader = copy.deepcopy(loader)
    [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y] = stats_list
    (mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y)=(mean_vec_x.to(device),
            std_vec_x.to(device),mean_vec_edge.to(device),std_vec_edge.to(device),mean_vec_y.to(device),std_vec_y.to(device))

    for data, viz_data, gs_data, eval_data in zip(loader, viz_data_loader,
                                                  gs_data_loader, eval_data_loader):
        data=data.to(args.device) 
        viz_data = data.to(args.device)
        with torch.no_grad():
            pred = best_model(data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge)
            # pred gives the learnt accelaration between two timsteps
            # next_vel = curr_vel + pred * delta_t  
            viz_data.x[:, 0:2] = data.x[:, 0:2] + pred[:]* delta_t
            gs_data.x[:, 0:2] = data.x[:, 0:2] + data.y* delta_t
            # gs_data - viz_data = error_data
            eval_data.x[:, 0:2] = (viz_data.x[:, 0:2] - gs_data.x[:, 0:2])
  
    #print(viz_data_loader)
    make_animation(gs_data_loader, viz_data_loader, eval_data_loader, file_dir,
                      gif_name, skip, True, False)

    return eval_data_loader

def draw_graph(g, save = False, args = None):
  G = to_networkx(g, to_undirected=True)
  pos = nx.spring_layout(G, seed=42)
  cent = nx.degree_centrality(G)
  node_size = list(map(lambda x: x * 500, cent.values()))
  cent_array = np.array(list(cent.values()))
  threshold = sorted(cent_array, reverse=True)[10]
  cent_bin = np.where(cent_array >= threshold, 1, 0.1)
  plt.figure(figsize=(12, 12))
  nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size,
                                cmap=plt.cm.plasma,
                                node_color=cent_bin,
                                nodelist=list(cent.keys()),
                                alpha=cent_bin)
  edges = nx.draw_networkx_edges(G, pos, width=0.25, alpha=0.3)
  if save and args is not None:
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    plt.title(f'Graph num nodes: {args.num_nodes}')
    plt.savefig(os.path.join(args.save_dir, f'graph_{args.num_nodes}'))
  
  plt.show()

@torch.no_grad()
def plot_mesh(gs, args):
  fig, ax = plt.subplots(1, 1, figsize=(20, 16))
  bb_min = gs.x[:, 0:2].min() # first two columns are velocity
  bb_max = gs.x[:, 0:2].max() # use max and min velocity of gs dataset at the first step for both 
                                    # gs and prediction plots


  ax.cla()
  ax.set_aspect('equal')
  ax.set_axis_off()

  pos = gs.mesh_pos
  faces = gs.cells
  velocity = gs.x[:, 0:2]


  triang = mtri.Triangulation(pos[:, 0].cpu(), pos[:, 1].cpu(), faces.cpu())
  mesh_plot = ax.tripcolor(triang, velocity[:, 0].cpu(), vmin= bb_min, vmax=bb_max,  shading='flat' ) # x-velocity
  ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)


  ax.set_title('SET', fontsize = '20')
  #ax.color

  #if (count == 0):
  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='5%', pad=0.05)
  clb = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
  clb.ax.tick_params(labelsize=20) 

  clb.ax.set_title('x velocity (m/s)',
                      fontdict = {'fontsize': 20})
  fig,
  return fig