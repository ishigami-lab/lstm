import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from env.env import GridMap
from env.utils import CraterProp, Param

def main():
    print("start!!")

    crater_prop = CraterProp(distribution="single",
                            geometry="mound",
                            num_crater=1,
                            min_D=100,
                            max_D=100)

    param = Param(n=250,
                res=1,
                re=0.8,
                sigma=10,
                is_fractal=False,
                is_crater=True,
                crater_prop=crater_prop)

    grid_map = GridMap(param=param)
    grid_map.set_terrain_env()

    grid_map.print_grid_map_info()
    grid_map.plot_maps(figsize=(8, 5))
    plt.show()
    
    print("done!!")


if __name__ == '__main__':
    main()