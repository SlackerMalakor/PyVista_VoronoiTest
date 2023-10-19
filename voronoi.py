import numpy as np
from pyvista import examples
import pyvista as pv
from scipy.spatial import Voronoi
import pyacvd

def generate_boundary_points(bounds, density = 20):
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    x = np.linspace(x_min, x_max, density)
    y = np.linspace(y_min, y_max, density)
    z = np.linspace(z_min, z_max, density)
    points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    is_on_boundary = np.logical_or.reduce([
        points[:,0] == x_min,
        points[:,0] == x_max,
        points[:,1] == y_min,
        points[:,1] == y_max,
        points[:,2] == z_min,
        points[:,2] == z_max
    ])
    boundary_points = points[is_on_boundary]
    
    return boundary_points

def generate_voronoi_lattice(mesh):    
    clustering = pyacvd.Clustering(mesh)
    clustering.cluster(mesh.n_cells)
    clustered_mesh = clustering.create_mesh()

    seed_points = np.vstack([clustered_mesh.points, generate_boundary_points(mesh.bounds)])
    vor = Voronoi(seed_points)

    lines = []
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            lines.append(np.append([len(simplex)], simplex))

    lines = np.concatenate(lines).flatten()
    return pv.PolyData(vor.vertices, lines).clip_surface(clustered_mesh).extract_largest()

mesh = examples.download_cow().triangulate()
voronoi_partitions = generate_voronoi_lattice(mesh)
voronoi_partitions["random_colors"] = np.random.rand(voronoi_partitions.n_cells, 3)

plotter = pv.Plotter()
plotter.add_mesh(voronoi_partitions, scalars = "random_colors", rgb = True)
plotter.show_grid()
plotter.show()