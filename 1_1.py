import pandas as pd
import numpy as np
import time


N_state = 6
Action = ['left','right']
Epsilon = 0.9
Alpha = 0.1
Gamma = 0.9

Max_episode = 13

fresh_time = 0.3

def build_q_table(n_state,actions):
    table = pd.DataFrame(np.zeros([n_state,len(actions)]),
                         columns=actions
                         )
    return table

def choose_action(state,q_table):
    state_action = q_table.iloc[state,:]
    if(np.random.uniform()>Epsilon) or (state_action.all()==0):
        action_name = np.random.choice(Action)
    else:
        action_name=state_action.idxmax()
    return action_name

def get_env_feedback(S,A):
    if A=='right':
        if S==N_state-2:
            S_='Terminal'
            R=1
        else:
            S_ = S+1
            R=0
    else:
        R=0
        if S==0:
            S_=S
        else:
            S_=S-1
    return S_,R

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_state-1) + ['T']   # '---------T' our environment
    if S == 'Terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')


def rl():
    q_table = build_q_table(N_state,Action)
    epsisode = 1
    while(1):
        step_counter =0
        S =0
        is_Terminal = False
        update_env(S,epsisode,step_counter)
        while not is_Terminal:
            A = choose_action(S,q_table)
            S_,R = get_env_feedback(S,A)
            temp = S_
            if S_ == 'Terminal':
                is_Terminal=True
                epsisode+=1
                temp = S_
                S_ = 5

            q_predict = q_table.iloc[S_,:].max()
            S_ = temp
            update_env(S_, epsisode, step_counter)
            #update q
            q_table[A][S] = q_table[A][S]+ Alpha*(R+Gamma*q_predict-q_table[A][S])
            S = S_
            step_counter += 1


if __name__ == '__main__':
    rl()