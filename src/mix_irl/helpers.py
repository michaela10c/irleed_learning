import sys, os
sys.path.append(os.path.abspath(os.path.join('../src')))
from mix_irl import trajectory as T
from mix_irl import irl as I

from irl_maxent import gridworld as W
from irl_maxent import solver as S

import numpy as np

# HELPER METHODS #
def setup_mdp(setting=0):
    """
    Set-up our MDP/GridWorld
    """
    # deterministc 
    if setting == 0:
        size = 5
    elif setting == 1:
        size = 7
    elif setting == 2:
        size = 9

    if setting in [0,1,2]:
        world = W.GridWorld(size=size)
        # set up the reward function
        reward = np.zeros(world.n_states)
        reward[size**2-1] = 1.0
        reward[size-1] = 1.0
        reward[0] = 1.0
        terminal = [0,size-1,size**2-1]
        # remove 0,4,24 from initial states
        initial = np.concatenate([np.arange(1,size-1),np.arange(size,size**2-1)])

    elif setting == 3:
        size = 4
        world = W.GridWorld(size=size)
        # set up the reward function
        reward = np.zeros(world.n_states)
        reward[size**2-1] = 1.0
        terminal = [size**2-2,size**2-1]
        # remove 0,4,24 from initial states
        initial = np.concatenate([np.arange(0,size),np.arange(1,size)*size])

    return world, reward, terminal, initial

def get_traj(world, theta, initial, terminal, horizon, n_traj, discount, weight=None):
    """
    Generate some trajectories, 
    weight determines the suboptimality level
    if weight is None will retern exper trajectory (deterministic)
    """
    if weight is None:
        # deterministic policy
        policy = S.optimal_policy(world, theta, discount, eps=1e-3)
        policy_exec = T.policy_adapter(policy)
        tjs = list(T.generate_trajectories(n_traj, world, policy_exec, initial, terminal, horizon))
    else:
        # stochastic policy
        weighting = lambda x: x**weight
        value = S.stochastic_value_iteration(world.p_transition, theta, discount)
        policy = S.stochastic_policy_from_value(world, value, w=weighting)
        policy_exec = T.stochastic_policy_adapter(policy)
        tjs = list(T.generate_trajectories(n_traj, world, policy_exec, initial, terminal, horizon))

    return tjs, policy

def get_mix_traj(world, initial, terminal, horizon, n_trajs, discount, features, true_reward, weights, epsilons):
    """
    Generate mixture of trajectories, 
    weights determines the suboptimality level and number of trajectories to generate
    if weight is None will retern expert trajectory (deterministic)

    Outputs: list of trajectory & policy class containing len(weights) items each
    """
    mix_traj = []
    mix_policy = []
    for weight, epsilon, n_traj in zip(weights, epsilons, n_trajs):
        reward = features.dot(true_reward+epsilon)
        traj, policy = get_traj(world, reward, initial, terminal, horizon, n_traj, discount, weight) 
        mix_traj.append(traj)
        mix_policy.append(policy)
    return mix_traj, mix_policy

def eval_traj(reward, trajectories):
    '''
    Evaluates reward of given trajectory

    Returns mean and std
    '''
    rews = []
    lens = []
    for t in trajectories:
        lens.append(len(t))
        rew = 0
        for s in t.states():
            rew += reward[s]
        rews.append(rew)
    rews = np.array(rews)
    lens = np.array(lens)

    return rews.mean(), lens.mean()

def eval_theta(setup, theta):
    '''
    Evaluate the current theta by sampling trajectories and computing their performance
    '''
    n_eval_traj = setup['n_e_traj']
    world = setup['world']
    terminal = setup['terminal']
    true_reward = setup['true_reward']
    initial = setup['initial']
    horizon = setup['horizon']
    discount = setup['discount']
    features = setup['features']
    
    # basic reward and policy eval
    reward = features.dot(theta)
    traj, policy = get_traj(world, reward, initial, terminal, horizon, n_eval_traj, discount)
    
    mean_rew, mean_len = eval_traj(true_reward, traj)

    return mean_rew, mean_len, traj, policy

def get_setup(ratios, weights, lam, n_traj, options, epsilons=None):
    setup = {}
    # save init paramters
    setup['world'], setup['true_reward'], setup['terminal'], setup['initial'] = setup_mdp(options['env_id'])
    setup['lam'] = lam
    setup['ratios'] = ratios
    setup['weights'] = weights

    fix_eps_zero = options.get('fix_eps_zero', False)

    
    # ---DEBUG: Include whether corners are included---
    size = setup['world'].size
    
    def idx(x, y):
        return setup['world'].state_point_to_index((x, y))
    
    def pt(s):
        return setup['world'].state_index_to_point(int(s))

    print("size:", size)
    print("terminal states:", setup['terminal'], "points:", [pt(s) for s in setup['terminal']])
    print("initial count:", len(setup['initial']))
    print("initial corners included?",
          {corner: (setup['world'].state_point_to_index(corner) in set(setup['initial']))
           for corner in [(0,0),(0,size-1),(size-1,0),(size-1,size-1)]})
    # ---END (DEBUG)---

    if epsilons is not None:
        # explicit epsilons passed in
        setup['epsilons'] = np.array(epsilons)
    elif fix_eps_zero:
        # force all per-demonstrator epsilons to zero
        setup['epsilons'] = np.zeros((len(ratios), setup['true_reward'].shape[0]))
    else:
        # original behavior: sample epsilons from N(0, I / lam^2), except for huge lam
        if lam > 20:
            setup['epsilons'] = np.zeros((len(ratios), setup['true_reward'].shape[0]))
        else:
            setup['epsilons'] = np.random.multivariate_normal(
                np.zeros_like(setup['true_reward']),
                np.eye(setup['true_reward'].shape[0]) / (lam**2),
                len(ratios)
            )
    
    setup['n_traj'] = n_traj
    setup['discount'] = options['discount'] 
    setup['horizon'] = options['horizon'] 
    setup['n_e_traj'] = options['n_e_traj']
    setup['n_s_traj'] = options['n_s_traj']
    setup['causal'] = options['causal']
    setup['options'] = options

    # create needed params
    setup['p_transition'] = setup['world'].p_transition
    setup['features'] = W.state_features(setup['world'])
    setup['n_states'], _, setup['n_actions'] = setup['p_transition'].shape

    setup['n'] = len(ratios)
    setup['n_trajs'] = [int(ratio*n_traj) for ratio in ratios]

    setup['mix_traj'], setup['mix_policy'] = get_mix_traj(setup['world'],  
                                                            setup['initial'], 
                                                            setup['terminal'], 
                                                            setup['horizon'], 
                                                            setup['n_trajs'], 
                                                            setup['discount'],
                                                            setup['features'], 
                                                            setup['true_reward'],
                                                            weights,
                                                            setup['epsilons'])

    # =========================
    # DEBUG: visitation + endings
    # =========================
    world = setup['world']

    # pick cells to analyze
    corners_xy = [(0,0), (size-1,0), (0,size-1), (size-1,size-1)]
    # 2 edges (non-corner)
    edges_xy   = [(0, size//2), (size-1, size//2)]
    # 2 interior
    interiors_xy = [(size//2, size//2), (size//2 - 1, size//2)]

    analyze_xy = corners_xy + edges_xy + interiors_xy
    analyze_ids = [idx(x,y) for (x,y) in analyze_xy]

    terminals_set = set(setup['terminal'])

    def per_traj_stats(state_idx, traj):
        """
        Returns:
          hit (0/1),
          count_visits (total visits to state in trajectory.states()),
          preterminal_count_visits (visits before first terminal is entered),
          ended_at_terminal (0/1),
          end_state (last state)
        """
        ss = list(traj.states())
        end_state = ss[-1]
        ended_at_terminal = int(end_state in terminals_set)

        # visits in whole trajectory
        c_total = sum(1 for s in ss if s == state_idx)
        hit = int(c_total > 0)

        # visits before termination (if terminal reached)
        if ended_at_terminal:
            # first time we enter any terminal
            first_term_t = next(i for i, s in enumerate(ss) if s in terminals_set)
            ss_pre = ss[:first_term_t]  # excludes the terminal entry itself
            c_pre = sum(1 for s in ss_pre if s == state_idx)
        else:
            c_pre = c_total

        return hit, c_total, c_pre, ended_at_terminal, end_state

    # Use component 0 for debugging (single-component case, or just inspect first component)
    trajs = setup['mix_traj'][0]
    N = len(trajs)

    # ---- End-state breakdown ----
    end_states = [list(t.states())[-1] for t in trajs]
    end_counts = {s: 0 for s in setup['terminal']}
    not_reach = 0
    for es in end_states:
        if es in terminals_set:
            end_counts[es] += 1
        else:
            not_reach += 1

    print("\n--- DEBUG: End-state breakdown (component 0) ---")
    for s in setup['terminal']:
        print(f"End at {pt(s)} (state {s}): {end_counts[s]/N:.3f} ({end_counts[s]}/{N})")
    print(f"Did NOT reach terminal: {not_reach/N:.3f} ({not_reach}/{N})")

    # ---- Hit-rate + dwell stats for selected states ----
    print("\n--- DEBUG: Hit-rate & dwell stats (component 0) ---")
    print("Note: 'preterm' means visits before first terminal is entered (or whole traj if no terminal).")

    for (x, y), s_id in zip(analyze_xy, analyze_ids):
        hits = 0
        dwell_total = []
        dwell_pre = []

        for t in trajs:
            hit, c_total, c_pre, ended_at_terminal, end_state = per_traj_stats(s_id, t)
            if hit:
                hits += 1
                dwell_total.append(c_total)
                dwell_pre.append(c_pre)

        hit_rate = hits / N
        if hits > 0:
            mean_total = float(np.mean(dwell_total))
            med_total  = float(np.median(dwell_total))
            max_total  = int(np.max(dwell_total))

            mean_pre = float(np.mean(dwell_pre))
            med_pre  = float(np.median(dwell_pre))
            max_pre  = int(np.max(dwell_pre))
        else:
            mean_total = med_total = max_total = 0.0
            mean_pre = med_pre = max_pre = 0.0

        tag = []
        if (x, y) in corners_xy: tag.append("CORNER")
        if s_id in terminals_set: tag.append("TERMINAL")
        if (x, y) in edges_xy: tag.append("EDGE")
        if (x, y) in interiors_xy: tag.append("INTERIOR")
        tag_str = ",".join(tag) if tag else "STATE"

        print(
            f"{tag_str:>16}  (x,y)=({x},{y}) state={s_id:2d} | "
            f"hit={hit_rate:.3f} | "
            f"dwell_total mean/med/max={mean_total:.2f}/{med_total:.1f}/{max_total} | "
            f"dwell_preterm mean/med/max={mean_pre:.2f}/{med_pre:.1f}/{max_pre}"
        )

    # ---- Optional: terminal-corner "reach" as end-state vs "hit at all" ----
    # (Sometimes you might *hit* a terminal only at the very end; this clarifies.)
    print("\n--- DEBUG: Terminal 'hit' vs 'end' (component 0) ---")
    for s in setup['terminal']:
        hit_any = 0
        end_here = 0
        for t in trajs:
            ss = list(t.states())
            if s in ss:
                hit_any += 1
            if ss[-1] == s:
                end_here += 1
        print(f"Terminal {pt(s)} state {s}: hit_any={hit_any/N:.3f}, end_here={end_here/N:.3f}")
    # =========================
    # END DEBUG
    # =========================

    
    # ---DEBUG---
    # Confirm "starting state bias"
    starts = [t.transitions()[0][0] for t in setup['mix_traj'][0]]
    unique, counts = np.unique(starts, return_counts=True)
    top = sorted(zip(counts, unique), reverse=True)[:10]
    print("Top start states:", [(c, pt(s)) for c, s in top])
    print("Start at (0,6) count:", np.sum(np.array(starts) == setup['world'].state_point_to_index((0, size-1))))


    # How long do we stay at (0, 6) once we hit it?
    corner = setup['world'].state_point_to_index((0, setup['world'].size - 1))

    dwell = []
    hits = 0
    for t in setup['mix_traj'][0]:
        ss = list(t.states())
        c = sum(1 for s in ss if s == corner)
        if c > 0:
            hits += 1
            dwell.append(c)

    print("Hit (0,6) rate:", hits / len(setup['mix_traj'][0]))
    if dwell:
        print("Mean #visits to (0,6) given hit:", np.mean(dwell), "median:", np.median(dwell), "max:", np.max(dwell))

    
    # Compare to non-corner ("control") cell, like (3, 3)
    ctrl = setup['world'].state_point_to_index((3,3))

    def cond_dwell(state_idx):
        dwell=[]
        hits=0
        for t in setup['mix_traj'][0]:
            ss=list(t.states())
            c=sum(1 for s in ss if s==state_idx)
            if c>0:
                hits += 1
                dwell.append(c)
        return hits/len(setup['mix_traj'][0]), (np.mean(dwell) if dwell else 0)

    hr_c, mean_c = cond_dwell(corner)
    hr_m, mean_m = cond_dwell(ctrl)
    print("corner hit, mean_dwell:", hr_c, mean_c)
    print("ctrl   hit, mean_dwell:", hr_m, mean_m)

    
    # sanity checks
    for k, trajectories in enumerate(setup['mix_traj']):
        lens = [len(t) for t in trajectories]
        ends = [list(t.states())[-1] for t in trajectories]
        reach = np.mean([e in setup['terminal'] for e in ends])
        print(f"[component {k}] n={len(trajectories)} mean_len={np.mean(lens):.2f} reach_terminal={reach:.3f}")
        print("  first traj states:", list(trajectories[0].states())[:15], "...")
    # ---END (DEBUG)---
    
    mix_e_features = []
    mix_p_initial = []
    rews, lens = [], []
    for trajectories in setup['mix_traj']:
        e_features = I.feature_expectation_from_trajectories(setup['features'], trajectories)
        p_initial = I.initial_probabilities_from_trajectories(setup['n_states'], trajectories)
        mix_e_features.append(e_features)
        mix_p_initial.append(p_initial)
        rew, length = eval_traj(setup['true_reward'],trajectories)
        rews.append(rew)
        lens.append(length)
    setup['mix_e_features'] = mix_e_features
    setup['mix_p_initial'] = mix_p_initial
    setup['dem_rews'] = np.mean(rews)
    setup['dem_lens'] = np.mean(lens)
    return setup
