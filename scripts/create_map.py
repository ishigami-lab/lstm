import matplotlib.pyplot as plt
from env.env import GridMap

def main():
    print("start!!")

    grid_map = GridMap(200, 1)
    grid_map.set_terrain_env(is_crater=True, is_fractal=True, num_crater=6)

    grid_map.print_grid_map_info()
    grid_map.plot_maps(figsize=(10, 4))
    plt.show()
    
    print("done!!")


if __name__ == '__main__':
    main()