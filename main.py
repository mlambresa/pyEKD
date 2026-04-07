import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from itertools import combinations, product
import multiprocessing as mp
import os
import time

########################################################
# PARAMETERS
#########################################################

N = 1000
#coupling_vec=np.logspace(-2,np.log10(300),21)
coupling_vec = np.array([10.0])
alpha_vec = np.logspace(-3,1,41)
mobility_vec= np.logspace(-2, 2, 41)
beta = 1
timestep = 0.01
T = 100
degree_average = np.array([6])
realizations=50


#########################################################
# EVOLUTION
#########################################################

def evolution(T, timestep, coupling, alpha, beta, r, footstep):
    #iteration over time
    G, nat_freq, phases, strategy, phase_dot, pos = initialize_TPG_PBC(N, r,coupling)
    R_G = []
    R_L = []
    C = [] 
    flag=0
    for t in np.arange(0,T,timestep):
        # Synchronization step
        phases = RK4(G, coupling, nat_freq, phases, strategy, timestep)
        # Payoffs update
        r_l = benefit(G,phases)  
        c, phase_dot = cost(G, coupling, phases, nat_freq, strategy, phase_dot)      
        payoff = update_payoff(G, alpha, r_l, c)
        # Game step
        strategy=update_strategy(G, beta, payoff, strategy)
        # Mobility step
        G, pos = random_mobility(G, pos, r, footstep)       
        # Collect data
        r_G = global_param(G,phases)
        r_L = local_param(G,r_l)
        coop = cooperation(G,strategy)
        R_G.append(r_G)
        R_L.append(r_L)
        C.append(coop)
        #exit condition
        if all(strategy[node] == 0 for node in G.nodes()) or all(strategy[node] == 1 for node in G.nodes()):
            flag+=1
            if flag==500:
                break
        #if all(strategy[node] == 1 for node in G.nodes() if G.degree(node) != 0) or all(strategy[node] == 0 for node in G.nodes() if G.degree(node) != 0):
        #    flag+=1
        #    if flag==100:
        #        break
    return R_G, R_L, C

#########################################################
# SYNCHRONIZATION STEP
#########################################################

def dynamics(G, coupling, nat_freq, phases, strategy):
    dthetadt = np.zeros(len(G))
    for node in G:
        if strategy[node] == 1 and G.degree(node) != 0:
            interactions = 0
            for neighbor in G.neighbors(node):
                interactions += np.sin(phases[neighbor] - phases[node])
            dthetadt[node] = nat_freq[node] + coupling * interactions
        else:
            dthetadt[node] = nat_freq[node]
    return dthetadt

def RK4(G, coupling, nat_freq, phases, strategy, timestep):
    K1 = dynamics(G, coupling, nat_freq, phases, strategy)
    phase1 = phases + timestep * K1 / 2
    
    K2 = dynamics(G, coupling, nat_freq, phase1, strategy)
    phase2 = phases + timestep * K2 / 2
    
    K3 = dynamics(G, coupling, nat_freq, phase2, strategy)
    phase3 = phases + timestep * K3
    
    K4 = dynamics(G, coupling, nat_freq, phase3, strategy)
    phase4 = phases + (timestep / 6) * (K1 + 2 * K2 + 2 * K3 + K4)
    
    return phase4

###########################################################
# GAME STEP
###########################################################

def benefit(G, phases):
    r_l = np.zeros(len(G.nodes))
    for node in G.nodes:
        if G.degree(node) != 0:
            r_lm = 0
            for neighbor in G.neighbors(node):
                r_lm += np.abs(np.cos((phases[node] - phases[neighbor]) / 2))
            r_l[node] = r_lm / G.degree(node)
        else:
            r_l[node] = 0
    return r_l

def cost(G, coupling, phases, nat_freq, strategy, phase_dot):
    phase_dot_new = dynamics(G, coupling, nat_freq, phases, strategy)
    c = np.zeros(len(G.nodes))
    for node in G.nodes:
        if strategy[node] == 1:
            c[node] = np.abs(phase_dot_new[node] - phase_dot[node])

    phase_dot[:] = phase_dot_new
    return c, phase_dot

def update_payoff(G, alpha, r_l, c):
    payoff = r_l - alpha / (2 * np.pi) * c
    return payoff

def update_strategy(G, beta, payoff, strategy):
    strategy_new = np.zeros(len(G.nodes))
    for node in G.nodes:
        if G.degree(node) != 0:
            neighbor = np.random.choice(list(G.neighbors(node)))
            if strategy[node] != strategy[neighbor]:
                prob = 1 / (1 + np.exp(- beta * (payoff[neighbor] - payoff[node])))
                if np.random.rand() < prob:
                    strategy_new[node] = strategy[neighbor]
                else :
                    strategy_new[node] = strategy[node]
            else :
                strategy_new[node] = strategy[node]
        else :
            strategy_new[node] = strategy[node]

    strategy[:] = strategy_new
    return strategy

#########################################################
# TEMPORAL PROXIMITY GRAPH
#########################################################

#RGG with PBC implementation
def toroidal_distance(point1, point2):
    dx = min(abs(point1[0] - point2[0]), 1 - abs(point1[0] - point2[0]))
    dy = min(abs(point1[1] - point2[1]), 1 - abs(point1[1] - point2[1]))
    return dx**2 + dy**2

def dx_dy(a,b):
    return abs(a[0] - b[0]), abs(a[1] - b[1])


def create_RGG(N):
    G = nx.empty_graph(N)
    pos = [[random.random(), random.random()] for _ in range(N)]
    return G, pos

def add_edges(G, pos, r):
    radius = r**2
    edges = []
    #boundary_edges=[]
    for u, v in combinations(range(len(pos)), 2):
        posu = pos[u]
        posv = pos[v]        
        d=toroidal_distance(posu,posv)
        dx,dy=dx_dy(posu,posv)
        if d <= radius and dx <= 0.5**2 and dy<=0.5**2:
            edges.append((u,v))
        elif d <= radius:
            edges.append((u,v))
            #boundary_edges.append((u,v))
    G.add_edges_from(edges)
    #edges_to_plot = [(u, v) for u, v in G.edges() if (u, v) not in boundary_edges]
    return G #edges_to_plot

def initialize_TPG_PBC(N,r,coupling):
    G, pos = create_RGG(N)
    G = add_edges(G, pos, r)

    nat_freq = np.random.uniform(-np.pi,np.pi,N)
    phase_init = np.random.uniform(-np.pi,np.pi,N)
    strategy_init = np.random.choice([0, 1], size=N)
    phase_dot_init = dynamics(G, coupling, nat_freq, phase_init, strategy_init)

    #nx.set_node_attributes(G, {node: freq for node, freq in zip(G.nodes(), nat_freq)}, "nat_freq")
    #nx.set_node_attributes(G, {node: phase for node, phase in zip(G.nodes(), phase_init)}, "phase")
    #nx.set_node_attributes(G, {node: s for node, s in zip(G.nodes(), strategy_init)}, "strategy")
    #nx.set_node_attributes(G, {node: pd for node, pd in zip(G.nodes(), phase_dot_init)}, "phase_dot")

    return G, nat_freq, phase_init, strategy_init, phase_dot_init, pos

#random mobility
def move_nodes(G, pos, footstep):
    for node in G.nodes():
        random_direction = np.random.uniform(0, 2*np.pi)
        new_x = pos[node][0] + footstep * np.cos(random_direction)
        new_y = pos[node][1] + footstep * np.sin(random_direction)

        pos[node][0] = new_x % 1
        pos[node][1] = new_y % 1


def random_mobility(G, pos, r,footstep):
    G.remove_edges_from(G.edges())
    move_nodes(G, pos, footstep)
    G=add_edges(G,pos,r)
    return G, pos

#########################################################
# ORDER PARAMETERS
##########################################################

def local_param(G, r_l):
    R_L = np.sum(r_l) / len(G.nodes)
    return R_L

def global_param(G, phases):
    complex_phases = np.exp(1j * np.array(phases))
    complex_phase = np.sum(complex_phases)
    return np.abs(complex_phase) / len(G.nodes)

def cooperation(G, strategy):
    C = np.sum(strategy)
    return C / len(G.nodes)

#########################################################
# MAIN 
#########################################################

def run_single_sim(params):
    alpha, coupling, mobility, degree_average, rep = params
    #np.random.seed(rep)  # for reproducibility
    r = np.sqrt(degree_average/(np.pi*N))

    footstep= r * mobility

    # check if file already exists
    filename = f"results/sim_lambda{coupling:.6f}_alpha{alpha:.6f}_mu{mobility:.3f}_k{degree_average}_rep{rep}.npz"
    if os.path.exists(filename):
        print(f"[SKIP] {filename} already exists, skipping simulation.")
        return filename

    start = time.perf_counter()   # start timer
    # call evolution
    R_G, R_L, C = evolution(T, timestep, coupling, alpha, beta, r, footstep)

    # save .npz file
    np.savez(filename, R_G=np.array(R_G), R_L=np.array(R_L), C=np.array(C))
    elapsed = time.perf_counter() - start  # end timer

    print(f"[PROC] {filename} saved after {elapsed:.2f} s")

    return filename

if __name__ == "__main__":
    # results directory
    os.makedirs("results", exist_ok=True)

    # simulation parameters
    param_grid = list(product(alpha_vec, coupling_vec, mobility_vec, degree_average, range(realizations)))

    # Parallelizzazione su tutte le CPU disponibili
    n_cpus = mp.cpu_count()
    print(f"Starting run on {n_cpus} CPU...")
    with mp.Pool(processes=n_cpus) as pool:
        all_files = pool.map(run_single_sim, param_grid)

    print("All simulations completed")

