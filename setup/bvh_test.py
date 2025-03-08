import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

def build_bvhtree_from_object(obj, depsgraph):
    """
    Build a BVH tree for the given object using its evaluated mesh.
    This function converts the mesh to a BMesh and then constructs the BVH tree.
    """
    # Get the evaluated object to include modifiers
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    
    # Create a BMesh from the mesh data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    
    # Build the BVH tree from the BMesh
    bvhtree = BVHTree.FromBMesh(bm)
    
    # Clean up: free the BMesh and clear the temporary mesh data
    bm.free()
    eval_obj.to_mesh_clear()
    
    return bvhtree

def test_ray_cast(bvhtree, start, direction, epsilon=1e-5, max_iterations=10):
    """
    Cast a ray repeatedly from the start point in the given direction.
    
    Each time an intersection is found, the start is advanced slightly beyond the hit.
    This returns a list of intersection details.
    """
    test_point = start.copy()
    intersections = []
    for i in range(max_iterations):
        result = bvhtree.ray_cast(test_point, direction)
        if result[0] is None:
            break
        hit_location, hit_normal, face_index, distance = result
        intersections.append({
            "hit_location": hit_location.copy(),
            "hit_normal": hit_normal.copy(),
            "face_index": face_index,
            "distance": distance
        })
        # Advance the test point slightly past the hit to avoid re-hitting the same face.
        test_point = hit_location + direction * epsilon
    return intersections

def build_bvhtree_from_object_world(obj, depsgraph):
    """
    Build a BVH tree for the given object using its evaluated mesh, 
    but transform the mesh vertices to world space so that scaling (and other transforms) are applied.
    
    Parameters:
      - obj: a Blender object (mesh type).
      - depsgraph: the dependency graph (from bpy.context.evaluated_depsgraph_get()).
    
    Returns:
      - bvhtree: a BVHTree constructed from the world-space mesh.
    """
    # Get the evaluated object (with modifiers applied)
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    
    # Create a BMesh from the mesh data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    
    # Transform each vertex to world space
    for vert in bm.verts:
        vert.co = obj.matrix_world @ vert.co
    
    # Build the BVH tree from the transformed BMesh
    bvhtree = BVHTree.FromBMesh(bm)
    
    # Clean up
    bm.free()
    eval_obj.to_mesh_clear()
    
    return bvhtree


def main():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    # Get or create a cube object.
    cube = bpy.data.objects.get("Cube")
    if cube is None:
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
        cube = bpy.context.active_object
    
    # Build the BVH tree for the cube.
    #bvhtree = build_bvhtree_from_object(cube, depsgraph)

    if cube is not None:
        bvhtree = build_bvhtree_from_object_world(cube, depsgraph)
    # Define a ray direction.
    ray_direction = Vector((1, 0, 0))
    
    # Test a point that is inside the cube.
    testpoint=(0,0,0.41)
    inside_point = Vector(testpoint)  # Cube centered at (0,0,0) with size 2 -> inside
    intersections_inside = test_ray_cast(bvhtree, inside_point, ray_direction)
    print("Intersections for inside point:",testpoint)
    for inter in intersections_inside:
        print("Hit at:", inter["hit_location"], 
              "Normal:", inter["hit_normal"], 
              "Face index:", inter["face_index"], 
              "Distance:", inter["distance"])
    
    # Test a point that is outside the cube.
    outside_point = Vector((1.1, 0, 0))  # Outside the cube.
    intersections_outside = test_ray_cast(bvhtree, outside_point, ray_direction)
    print("Intersections for outside point (1.1,0,0)")
    for inter in intersections_outside:
        print("Hit at:", inter["hit_location"], 
              "Normal:", inter["hit_normal"], 
              "Face index:", inter["face_index"], 
              "Distance:", inter["distance"])
    
    # Optionally, you can visualize the ray(s) using the create_ray function (see previous examples).

if __name__ == "__main__":
    main()