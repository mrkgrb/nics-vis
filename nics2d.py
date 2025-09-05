# -*- coding: utf-8 -*-
"""
This script generates coordinates and Gaussian input files for NICS(h)zz 2D calculations (NICS maps) 
for all molecular *.xyz files in the working directory. The file template.gjf is used to provide the 
Gaussian keywords and settings. 

The script is designed for bent and twisted π-systems: it automatically identifies the longest fused 
polycyclic framework in each molecule and creates grids of Bq points elevated by h above and below the 
ring planes, perpendicular to them. 

After the Gaussian calculations, use the companion script nics_map.py to extract the results and produce 
visual 2D NICS maps.

For usage:
python nics2d.py --help
"""

import numpy as np
import networkx as nx
from scipy.spatial.transform import Rotation as R
import argparse
import os
import glob

def parse_xyz(file_path):
    """Parse an XYZ file into atomic symbols and coordinates."""
    atoms = []
    coords = []
    with open(file_path, 'r') as f:
        print(f"Loaded {file_path}")
        for line in f:
            parts = line.split()
            if len(parts) == 4:  # Expecting atomic symbol and x, y, z coordinates
                atoms.append(parts[0])
                coords.append(list(map(float, parts[1:4])))
    return atoms, np.array(coords)


def construct_graph(atoms, coords, bond_threshold=1.9):
    """Construct a molecular graph from atomic coordinates."""
    G = nx.Graph()
    num_atoms = len(atoms)
    for i in range(num_atoms):
        if atoms[i] != 'H':
            G.add_node(i, element=atoms[i], pos=coords[i])
            for j in range(i + 1, num_atoms):
                if atoms[j] != 'H':
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist <= bond_threshold:  # Adjust bond length tolerance
                        G.add_edge(i, j)
    return G


def find_non_hydrogen_rings(graph, atoms):
    """Identify rings formed only by non-hydrogen atoms."""
    all_rings = nx.cycle_basis(graph)
    non_hydrogen_rings = [
        ring for ring in all_rings if all(atoms[i] != 'H' for i in ring)
    ]
    return non_hydrogen_rings


def filter_fused_ring_systems(graph, rings):
    """Group rings into fused systems."""
    fused_graph = nx.Graph()
    for i, ring1 in enumerate(rings):
        for j, ring2 in enumerate(rings):
            if i < j:
                shared_edges = [
                    tuple(sorted((ring1[k], ring1[(k + 1) % len(ring1)])))
                    for k in range(len(ring1))
                ]
                if any(
                    tuple(sorted((ring2[k], ring2[(k + 1) % len(ring2)]))) in shared_edges
                    for k in range(len(ring2))
                ):
                    fused_graph.add_edge(i, j)
    connected_components = list(nx.connected_components(fused_graph))
    longest_system = max(
        connected_components, key=lambda comp: len(comp)
    )  # Longest fused system
    return [rings[i] for i in longest_system]


def calculate_mean_plane(coords):
    """Calculate the mean plane of a set of points using SVD."""
    centroid = np.mean(coords, axis=0)
    _, _, vh = np.linalg.svd(coords - centroid)
    normal = vh[2]
    return centroid, normal


def rotate_to_xy_plane(coords, normal):
    """Rotate the molecule such that the specified normal aligns with the Z-axis."""
    z_axis = np.array([0, 0, 1])
    rotation_vector = np.cross(normal, z_axis)
    if np.linalg.norm(rotation_vector) < 1e-6:
        return coords  # Already aligned
    rotation_angle = np.arccos(np.dot(normal, z_axis) / np.linalg.norm(normal))
    rotation_matrix = R.from_rotvec(rotation_vector * rotation_angle / np.linalg.norm(rotation_vector))
    rotated_coords = rotation_matrix.apply(coords)
    return rotated_coords

def align_principal_axis_to_x(centroids, coords):
    """Align the principal axis of the centroids to the X-axis."""
    # Calculate principal axis of the centroids
    centered_centroids = centroids - np.mean(centroids, axis=0)
    cov_matrix = np.cov(centered_centroids, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    principal_axis = eigenvectors[:, -1]  # Principal axis (largest eigenvalue)
    
    # Rotate the molecule to align the principal axis with the X-axis
    x_axis = np.array([1, 0, 0])
    rotation_vector = np.cross(principal_axis, x_axis)
    if np.linalg.norm(rotation_vector) > 1e-6:  # Avoid division by zero
        rotation_angle = np.arccos(np.dot(principal_axis, x_axis) / np.linalg.norm(principal_axis))
        rotation_matrix = R.from_rotvec(rotation_vector * rotation_angle / np.linalg.norm(rotation_vector))
        rotated_coords = rotation_matrix.apply(coords)
    else:
        rotated_coords = coords  # Already aligned with the X-axis
    return rotated_coords

def align_molecule_to_xy_and_x(principal_coords,coords):
    """
    Align the molecule's principal coordinates:
    - Align the mean plane of the largest polycyclic system to the XY plane.
    - Align the principal axis of the molecule to the X axis.
    """

    # Step 1: Calculate the mean plane normal using SVD
    centroid_mean = np.mean(principal_coords, axis=0)
    centered_coords = principal_coords - centroid_mean
    _, _, vh = np.linalg.svd(centered_coords, full_matrices=False)
    normal_vector = vh[-1]  # Normal to the mean plane (smallest singular vector)

    centroid_mean_all = np.mean(coords, axis=0)
    centered_coords_all = coords - centroid_mean_all
    
    # Ensure the normal vector points upwards (positive z-direction)
    if normal_vector[2] < 0:
        normal_vector = -normal_vector

    # Step 2: Rotate the molecule to align the mean plane with the XY plane
    z_axis = np.array([0, 0, 1])
    axis_of_rotation = np.cross(normal_vector, z_axis)
    angle = np.arccos(np.dot(normal_vector, z_axis))
    if np.linalg.norm(axis_of_rotation) > 1e-6:  # Avoid division by zero
        axis_of_rotation /= np.linalg.norm(axis_of_rotation)
        rotation_matrix = rotation_matrix_from_axis_angle(axis_of_rotation, angle)
        aligned_coords = centered_coords @ rotation_matrix.T
        aligned_coords_all = centered_coords_all @ rotation_matrix.T
    else:
        aligned_coords = centered_coords  # Already aligned with z-axis
        aligned_coords_all = centered_coords_all  # Already aligned with z-axis

    # Step 3: Align the principal axis to the X axis
    cov_matrix = np.cov(aligned_coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    principal_axis = eigvecs[:, -1]  # Principal eigenvector (largest eigenvalue)

    # Ensure the principal axis points in the positive x-direction
    if principal_axis[0] < 0:
        principal_axis = -principal_axis

    x_axis = np.array([1, 0, 0])
    axis_of_rotation = np.cross(principal_axis, x_axis)
    angle = np.arccos(np.dot(principal_axis, x_axis))
    if np.linalg.norm(axis_of_rotation) > 1e-6:
        axis_of_rotation /= np.linalg.norm(axis_of_rotation)
        rotation_matrix = rotation_matrix_from_axis_angle(axis_of_rotation, angle)
        aligned_coords = aligned_coords @ rotation_matrix.T
        aligned_coords_all = aligned_coords_all @ rotation_matrix.T

    return aligned_coords_all

def rotation_matrix_from_axis_angle(axis, angle):
    """
    Generate a rotation matrix for a given axis and angle using Rodrigues' formula.
    """
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    identity = np.eye(3)
    return identity + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)



def calculate_ring_centroids_and_normals(rings, coords):
    """Calculate centroids and normals for the given rings."""
    centroids = []
    normals = []
    for ring in rings:
        ring_coords = coords[ring]
        # Calculate centroid
        centroid = np.mean(ring_coords, axis=0)
        # Calculate normal vector using SVD
        _, _, vh = np.linalg.svd(ring_coords - centroid)
        normal = vh[2]
        
        centroids.append(centroid)
        normals.append(normal)
    return centroids, normals


def ensure_normals_consistent(normals, reference_normal):
    """Ensure all normals point in the same direction."""
    consistent_normals = []
    for normal in normals:
        if np.dot(normal, reference_normal) < 0:
            consistent_normals.append(-normal)
        else:
            consistent_normals.append(normal)
    return consistent_normals


def calculate_shared_edge_centroids_and_normals(rings, coords, ring_normals):
    """Calculate centroids and normals for shared edges."""
    edges = set()
    shared_edge_centroids = []
    shared_edge_normals = []
    for i, ring in enumerate(rings):
        for j, atom1 in enumerate(ring):
            atom2 = ring[(j + 1) % len(ring)]  # Next atom in the ring
            edge = tuple(sorted((atom1, atom2)))
            if edge in edges:
                # Calculate edge centroid
                edge_coords = coords[list(edge)]
                centroid = np.mean(edge_coords, axis=0)
                # Interpolate normals for the edge center between two connected rings
                for k, other_ring in enumerate(rings):
                    if k != i and edge in {
                        tuple(sorted((other_ring[m], other_ring[(m + 1) % len(other_ring)])))
                        for m in range(len(other_ring))
                    }:
                        edge_normal = (ring_normals[i] + ring_normals[k]) / 2
                        edge_normal /= np.linalg.norm(edge_normal)
                        shared_edge_centroids.append(centroid)
                        shared_edge_normals.append(edge_normal)
                        break
            else:
                edges.add(edge)
    return shared_edge_centroids, shared_edge_normals

def project_onto_ZX_along_Y(p, lY, gY):
    
    t = - np.dot(gY, p) / np.dot(gY, lY)
    pr_vec = p + t * lY
    
    return pr_vec

def project_onto_plane(v, u1, u2):
    # Normalize the principal vectors
    u1_hat = u1 / np.linalg.norm(u1)
    u2_hat = u2 / np.linalg.norm(u2)

    # Calculate projections onto each vector
    proj_u1 = np.dot(v, u1_hat) * u1_hat
    proj_u2 = np.dot(v, u2_hat) * u2_hat

    # Sum the projections to get the projection onto the plane
    return proj_u1 + proj_u2

def interpolate_path(centroids, normals, d, h, x_ext, y_ext):
    """Interpolate a path with points elevated by h and spaced by d."""
    # Elevate points using normals
    zero_points = []
    dropped_points = []
    elevated_points = []
    zero_points = [c for c, n in zip(centroids, normals)]
    elevated_points = [c + h * n for c, n in zip(centroids, normals)]
    dropped_points = [c - h * n for c, n in zip(centroids, normals)]
    e_normals = [n for c, n in zip(centroids, normals)]

    
    # Calculate principal axis from covariance matrix
    centered = zero_points - np.mean(zero_points, axis=0)
    offset_mean = np.mean(zero_points, axis=0)
    principal_axis = [1,0,0]
    
    # Sort points along the principal axis
    projections = np.dot(centered, principal_axis)
    sorted_indices = np.argsort(projections)
    zero_points = np.array(zero_points)[sorted_indices]
    elevated_points = np.array(elevated_points)[sorted_indices]
    dropped_points = np.array(dropped_points)[sorted_indices]
    e_normals = np.array(e_normals)[sorted_indices]

    
    # Extend path by adding extra points at the beginning and end
    if len(zero_points) > 1:
        
        
        extension_vector_start = zero_points[0] - zero_points[1]
        extension_vector_end = zero_points[-1] - zero_points[-2]
        
        extension_vector_start = np.cross(e_normals[0], extension_vector_start)
        extension_vector_start = np.cross(extension_vector_start, e_normals[0])
        extension_vector_start = extension_vector_start / np.linalg.norm(extension_vector_start)
        extension_vector_end = np.cross(e_normals[-1], extension_vector_end)
        extension_vector_end = np.cross(extension_vector_end, e_normals[-1])
        extension_vector_end = extension_vector_end / np.linalg.norm(extension_vector_end)        
       
        # Step 3: Calculate new points extended by 3Å
        # Step 4: Append the new points to the array
        start_extension = elevated_points[0] + x_ext * extension_vector_start
        end_extension = elevated_points[-1] + x_ext * extension_vector_end
        
        start_extension = dropped_points[0] + x_ext * extension_vector_start
        end_extension = dropped_points[-1] + x_ext * extension_vector_end
        
        start_extension = zero_points[0] + x_ext * extension_vector_start
        end_extension = zero_points[-1] + x_ext * extension_vector_end
        ext_z_points = np.vstack([start_extension, zero_points, end_extension])
        
        ext_normals = np.vstack([e_normals[0], e_normals, e_normals[-1]])

    avg_normal = [0,0,1]
        
    glob_Y = np.cross(avg_normal,principal_axis)  
    y_width = y_ext
    num_points_y = int(2 * y_width / d)
    sqrt_3 = d * 0.57735 /2
    
    # Interpolation logic: create a dense path
    dense_path = []
    bq_xy_coords = []
    x_cor = 0.0
    for i in range(len(ext_z_points) - 1):
        start, end = ext_z_points[i] - offset_mean, ext_z_points[i + 1] - offset_mean
        start_normal, end_normal = ext_normals[i], ext_normals[i + 1]
        
        #project start end points on XZ plane 
        start_Y, end_Y = np.cross(start_normal,principal_axis), np.cross(end_normal,principal_axis)
        start_Y = start_Y / np.linalg.norm(start_Y)
        end_Y = end_Y / np.linalg.norm(end_Y)
        start = project_onto_ZX_along_Y(start, start_Y, glob_Y) + offset_mean
        end = project_onto_ZX_along_Y(end, end_Y, glob_Y) + offset_mean       
              
    
        segment_vector = end - start
        segment_length = np.linalg.norm(segment_vector)
        num_points = int(segment_length // d) + 1
        
        end_t = float(num_points -1)/float(num_points)
        if i == (len(ext_z_points) - 2):
            end_t = 1
            num_points += 1
        
        for t in np.linspace(0, end_t, num_points):
            local_normal = t * end_normal + (1-t) * start_normal
            local_normal = local_normal / np.linalg.norm(local_normal)
            local_y_axis = np.cross(local_normal, principal_axis)
            local_y_axis = local_y_axis/np.linalg.norm(local_y_axis)
            sqrt_3 = -sqrt_3
            
            for yp in np.linspace(-1, 1, num_points_y):
                y_cor = y_width * yp + sqrt_3
                dense_path.append(start + h * local_normal + t * segment_vector + (y_width * yp + sqrt_3) * local_y_axis)
                dense_path.append(start - h * local_normal + t * segment_vector + (y_width * yp + sqrt_3) * local_y_axis)
                bq_xy_coords.append([x_cor + t * segment_length,y_cor])
        x_cor += segment_length
    return np.array(dense_path), np.array(bq_xy_coords)


def generate_path_from_xyz(xyz_file, d, h, output_file, x_ext, y_ext):
    """Generate a path based on XYZ coordinates."""
    # Parse the XYZ file
    atoms, coords = parse_xyz(xyz_file)
    
    # Construct the molecular graph
    graph = construct_graph(atoms, coords, bond_threshold=1.9)
    
    # Find rings in the molecular graph
    rings = find_non_hydrogen_rings(graph, atoms)
    fused_ring_systems = filter_fused_ring_systems(graph, rings)
    
    if not fused_ring_systems:
        print("No suitable fused ring system detected.")
        return
    
    # Consider only the longest fused system
    longest_fused_system = fused_ring_systems
    
    # Calculate mean plane and rotate molecule
    system_coords = np.concatenate([coords[ring] for ring in longest_fused_system])
    rotated_coords = align_molecule_to_xy_and_x(system_coords,coords) # rotate_to_xy_plane(coords, mean_normal)
    # After rotating to the XY plane, align the principal axis to the X-axis:

    mean_normal = [0,0,1]
    
    # Calculate centroids and normals for the rings
    ring_centroids, ring_normals = calculate_ring_centroids_and_normals(longest_fused_system, rotated_coords)
    ring_normals = ensure_normals_consistent(ring_normals, mean_normal)
    
    # Calculate shared edge centroids and normals
    edge_centroids, edge_normals = calculate_shared_edge_centroids_and_normals(longest_fused_system, rotated_coords, ring_normals)
    edge_normals = ensure_normals_consistent(edge_normals, mean_normal)
    
    # Combine ring and edge centroids/normals
    all_centroids = ring_centroids + edge_centroids
    all_normals = ring_normals + edge_normals
    
    # Generate the path
    path_coordinates, xy_projections = interpolate_path(all_centroids, all_normals, d, h, x_ext, y_ext)
    output_file2 = output_file
    output_file2 = output_file2.replace("NMP.xyz", "NMP_XY.txt")
    
    # Write the output
    with open(output_file, "w") as f:
        f.write(f"{len(atoms) + len(path_coordinates)}\n")
        f.write("Molecule with interpolated path points\n")
        for atom, coord in zip(atoms, rotated_coords):
            f.write(f"{atom} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n")
        for point in path_coordinates:
            f.write(f"Bq {point[0]:.8f} {point[1]:.8f} {point[2]:.8f}\n")
            
    with open(output_file2, "w") as f:       
        for point in xy_projections:
            f.write(f"{point[0]:.6f} {point[1]:.6f}\n")
    print(f"Path and molecule saved to {output_file}.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate input files for NICS(h)zz 2D calculations.")
    parser.add_argument("--d", type=float, default=0.2, help="Interpoint distance between dummy atoms Bq (default: 0.2).")
    parser.add_argument("--h", type=float, default=1.0, help="Height of points above/below the system (default: 1.0).")
    parser.add_argument("--x", type=float, default=3.0, help="Extension of the map along x axis (default: 3.0, min: 2.0).")
    parser.add_argument("--y", type=float, default=4.0, help="Extension of the map along y axis (default: 4.0, min: 2.0).")
    parser.add_argument("--triplet", dest="trip", action="store_true", help="Triplet state?")
    parser.set_defaults(trip=False)
    args = parser.parse_args()
    
    x_ext = args.x if args.x >= 2.0 else 2.0
    y_ext = args.y if args.y >= 2.0 else 2.0
    
    print(f"Distance interval: {args.d:.3f}, Bq height: {args.h:.3f}, X and Y extensions: {x_ext:.1f}, {y_ext:.1f}")
    
    output_folder = "xyzs"
    os.makedirs(output_folder, exist_ok=True)
     # Load the Gaussian template from the file
    s0_t1 = "1"
    st_str = ""
    if args.trip:
        s0_t1 = "3"
        st_str = "t"

    # Process all *.xyz files in the folder
    xyz_files = glob.glob("*.xyz")
    for xyz_file in xyz_files:
        # Extract molecule name and solvent abbreviation from the filename
        base_name = os.path.splitext(xyz_file)[0]
        parts = base_name.split("-")
        molecule_name = parts[0]
        solvent_abbr = parts[1] if len(parts) > 1 else "Vac"  # Default solvent is Vac

        # Map solvent abbreviation to full name
        solvent_map = {
        "DCM": "dichloromethane",
        "ACN": "acetonitrile",
        "Hex": "n-hexane",
        "H2O": "water",
        "MeOH": "methanol",
        "Tol": "toluene",
        "Vac": "vacuum"
        }
        solvent_full_name = solvent_map.get(solvent_abbr, "vacuum")  # Default full name
        if solvent_full_name == "vacuum":
            solvent_abbr = "Vac"
            
        if solvent_abbr != "Vac":
            scrf_input = f" scrf=(smd,solvent={solvent_full_name})"
            mol_solvent = f"{molecule_name}-{solvent_abbr}"
        else:
            scrf_input = ""
            molecule_name = base_name
            mol_solvent = base_name
        
        output_xyz_file = os.path.join(output_folder, f"{mol_solvent}-{st_str}NMP.xyz")

        with open("template.gjf", "r") as template_file:
            gjf_template = template_file.read()

        #gjf_template
        generate_path_from_xyz(xyz_file, args.d, args.h, output_xyz_file, x_ext, y_ext)
       
        # Read the coordinates and assign consecutive atom numbers
        with open(output_xyz_file, "r") as f:
            lines = f.readlines()
        
        num_atoms_and_points = len(lines) - 2  # Exclude the XYZ header lines
        coords = "".join(lines[2:])  # Skip the first two header lines
        atom_numbers = "\n".join(str(i + 1) for i in range(num_atoms_and_points))
        
        calc_conditions = f"NICS 2D Map, Interpoint dist: {args.d:.3f}, Bq height: {args.h:.2f}"
        calc_type = f"{st_str}NMP"
        
        # Populate the template
        gjf_content = gjf_template.format(
            cores = "24",
            memory = "48gb",
            name_slv = mol_solvent,
            name=molecule_name,
            slv=solvent_abbr,
            slv_name=solvent_full_name,
            scrf_in=scrf_input,
            trip = s0_t1,
            calc_type = calc_type,
            calc_data = calc_conditions
        ).replace("coords", coords).replace("numbers", atom_numbers)
        
        # Define the Gaussian input file path
               
        output_gjf_file = f"{mol_solvent}-{calc_type}.gjf"
        
        # Write the Gaussian input file
        with open(output_gjf_file, "w") as f:
            f.write(gjf_content)
            print(f"Saved input file {output_gjf_file}.\n")
  