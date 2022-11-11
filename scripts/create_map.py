import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt

# import original structures and classes
from env.env import GridMap
from env.utils import CraterProp, Param
from env.surface_model import SurfaceModel

def main():
    print("start!!")

    crater_prop = CraterProp(distribution="random",
                            geometry="normal",
                            num_crater=50,
                            min_D=5,
                            max_D=10)

    param = Param(n=200,
                res=1,
                re=0.8,
                sigma=5,
                is_fractal=True,
                is_crater=True,
                crater_prop=crater_prop)

    grid_map = SurfaceModel(param=param)
    grid_map.set_terrain_env()

    grid_map.print_grid_map_info()
    grid_map.plot_maps()
    plt.show()
    
    print("done!!")


if __name__ == '__main__':
    main()