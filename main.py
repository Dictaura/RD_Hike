import math
import os
import random
import time

import numpy
from tqdm import tqdm

from rl_lib.environment import RNA_Env
import pathos.multiprocessing as pathos_mp
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def main():
    ############ initial parameter ############
    root = os.path.dirname(os.path.realpath(__file__))

    # 当前时间
    local_time = str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

    ###################### logging ######################

    log_dir_root = root + "/logs/"
    if not os.path.exists(log_dir_root):
        os.makedirs(log_dir_root)

    log_dir = log_dir_root + local_time
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir, comment="Train_Log.log")

    # 进程池
    pool_main = pathos_mp.ProcessPool()
    pool_env = pathos_mp.ProcessPool()

    # 环境
    aim_dotB = '...((((((((...)))))((.........))................((((...(((((((((((....))))))..)))))...)))).(((..((.(((....(((((..((....)).)))))....)))...))..)))..(((((..(.((((....)))))..)))))(((((.......)))))....((((((((.(((.((((((((....)))))))))))..))))).))).....))).'
    done_list = [
        'UCCCUGGGAGUAACACUCCCCGUAAAAAUAGGAACACAGCUAGGUCUAGGAUAUGCCCGGCAAUAGAUGCCUAUUGGACCGGGAGAAUCCACACGAGUAAUCUGCGGUCCCCUCCUAACGGUGGGACGAUUGAUACAACAAGUGAAGGCACUAGAGCUGCCGUCAGCCCCGUGCCCGUUUCCCGUCAAAACGCCGACACACGUUAUAUGUGGUGAUCUCGAGAUCACCAAUAGCAACGUAGUGGUAAACAGA'
    ]
    action_space = 4
    env = RNA_Env(aim_dotB, action_space, pool_env, done_list=done_list)

    # 运行设置
    # 最大步数
    max_step = 9999

    # 随机概率
    r_start = 1.
    r_end = 0.
    decay = 2500

    x = []
    y = []
    sol_cnt = 0
    consum_step = 0

    while(consum_step < max_step):

        env.reset()

        for t in tqdm(range(max_step - consum_step)):
            consum_step += 1
            x.append(t)
            y.append(env.last_distance)

            writer.add_scalar('distance_{}'.format(sol_cnt+1), env.last_distance, t)

            if env.last_distance == 0:
                sol_cnt += 1
                print("Done! sequence: {}, step: {}".format(env.seq_base, t+1))
                writer.add_text("solution_{}".format(sol_cnt), env.seq_base)
                writer.add_scalar('solve_step', t+1, sol_cnt)
                env.done_list.append(env.seq_base)
                break

            sample_ratio = r_end + (r_start - r_end) * math.exp(-1. * t / decay)
            near_seq_list, actions = env.get_near_seq()
            near_dist_list = env.get_near_distance(near_seq_list)
            near_energy_list = env.get_near_energy(near_seq_list)
            near_novel_list = env.get_near_novelty(near_seq_list)

            x = list(range(len(near_seq_list)))
            y1 = list(map(abs, numpy.array(near_energy_list)-env.get_energy(env.seq_base)))
            y2 = list(map(abs, numpy.array(near_dist_list) - env.last_distance))

            y1_min = min(filter(lambda x: x>0, y1))
            y2_min = min(filter(lambda x: x>0, y2))

            y1 = y1 / y1_min
            y2 = y2 / y2_min

            ax1 = plt.subplot(121)
            ax1.set_title("near energy")
            plt.bar(x, y1)
            ax2 = plt.subplot(122)
            ax2.set_title("near distance")
            plt.bar(x, y2)
            plt.show()


            r = random.random()

            type = 'max'

            if r < sample_ratio:
                type = 'sample'

            next_order = env.get_next_seq_order(near_dist_list, near_novel_list, type)

            next_seq = near_seq_list[next_order]
            action = actions[next_order]
            next_dist = near_dist_list[next_order]
            next_novel = near_novel_list[next_order]

            env.renew_env(next_seq, action, next_dist, next_novel)



        # plt.plot(x, y)
        # plt.show()

if __name__ == "__main__":
    main()





