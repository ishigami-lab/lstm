import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from env.env import GridMap
from env.utils import Param

def main():
    print("start!!")

    param = Param(n=200,
                res=1,
                re=0.8,
                sigma=10)

    grid_map = GridMap(param=param)
    grid_map.set_terrain_env(is_crater=True, is_fractal=True, num_crater=6)

    grid_map.print_grid_map_info()
    grid_map.plot_maps(figsize=(10, 4))
    plt.show()
    
    print("done!!")


if __name__ == '__main__':
    main()