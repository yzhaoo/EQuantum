import bpy
import json
import mathutils
from mathutils.bvhtree import BVHTree
import bmesh
import os
import sys
import time

def is_point_inside_object(point_vec, bvhtree, obj_matrix_world):
    """
    Test if a point is inside a closed mesh object using a ray-cast and BVH.
    """
    # Transform point to object space if BVH is in object space, 
    # BUT here we built BVH in world space for performance in one go? 
    # actually, building BVH for each object is fine if cached.
    # The original script built BVH in world space for each query? No, it built it once per object per call?
    # Original:
    # eval_obj = obj.evaluated_get(depsgraph)
    # mesh = eval_obj.to_mesh()
    # ... build BVH ...
    # This was done INSIDE the loop for every point! That's extremely slow.
    
    # Improved approach: pass the (bvhtree, world_matrix) or just bvhtree if it's world space.
    
    # Ray cast direction
    direction = mathutils.Vector((1, 1, 1))
    test_point = point_vec.copy()
    epsilon = 1e-5
    count = 0
    
    while True:
        result = bvhtree.ray_cast(test_point, direction)
        if result[0] is None:
            break
        hit_location, hit_normal, face_index, distance = result
        count += 1
        # Advance slightly
        test_point = hit_location + direction * epsilon

    return (count % 2) == 1

def get_bbox(obj):
    coords = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    xs = [p.x for p in coords]
    ys = [p.y for p in coords]
    zs = [p.z for p in coords]
    return {
        "xmin": min(xs), "xmax": max(xs),
        "ymin": min(ys), "ymax": max(ys),
        "zmin": min(zs), "zmax": max(zs)
    }

def point_in_bbox(point_list, bbox):
    x, y, z = point_list
    return (bbox["xmin"] <= x <= bbox["xmax"] and
            bbox["ymin"] <= y <= bbox["ymax"] and
            bbox["zmin"] <= z <= bbox["zmax"])

def main():
    start_time = time.time()
    
    # Args: blender file.blend --background --python assign_mat_optimized.py -- site_file output_file
    # But usually just run in the directory.
    # We'll use the same logic as original but optimized.
    
    filepath = bpy.data.filepath
    directory = os.path.dirname(filepath)
    sitefile = os.path.join(directory, "sites.json")
    outfile = os.path.join(directory, "setup_config1.json") # Saving to setup_config1.json as requested
    
    if not os.path.exists(sitefile):
        print(f"Error: {sitefile} not found.")
        return

    print(f"Loading sites from {sitefile}...")
    with open(sitefile, "r") as f:
        sites = json.load(f)
        
    print(f"Loaded {len(sites)} sites.")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    # Pre-process objects: Build BVH trees ONCE
    mesh_objects = []
    plane_obj = bpy.data.objects.get("2dplane")
    
    # Handle other objects
    # We will store (obj, bbox, bvhtree) tuples
    scene_objects = [obj for obj in bpy.context.scene.objects 
                     if obj.type == 'MESH' and obj.name != "2dplane"]
    
    print(f"Building BVH trees for {len(scene_objects)} objects...")
    
    prepared_objects = []
    
    for obj in scene_objects:
        # Get evaluated mesh
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        
        # Transform vertices to world space
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        for vert in bm.verts:
            vert.co = obj.matrix_world @ vert.co
        
        bvhtree = BVHTree.FromBMesh(bm)
        bm.free()
        eval_obj.to_mesh_clear()
        
        bbox = get_bbox(obj)
        prepared_objects.append({
            "obj": obj,
            "bbox": bbox,
            "bvh": bvhtree
        })
        
    print("BVH trees built.")

    # 2dplane special handling (bbox only, as per original script)
    if plane_obj:
        print("Processing 2dplane...")
        plane_bbox = {
            "xmin": min([p[0] for p in [plane_obj.matrix_world @ mathutils.Vector(c) for c in plane_obj.bound_box]]),
            "xmax": max([p[0] for p in [plane_obj.matrix_world @ mathutils.Vector(c) for c in plane_obj.bound_box]]),
            "ymin": min([p[1] for p in [plane_obj.matrix_world @ mathutils.Vector(c) for c in plane_obj.bound_box]]),
            "ymax": max([p[1] for p in [plane_obj.matrix_world @ mathutils.Vector(c) for c in plane_obj.bound_box]]),
            "zmin": 0.0, "zmax": 0.0 # original script hardcoded z limits for plane?
        }
        # Actually original script calculated full 3d bbox but then forced zmin/zmax to 0?
        # logic: 
        # coords = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        # zmin=0, zmax=0. 
        # Let's replicate original logic exactly for 2dplane.
        plane_coords = [plane_obj.matrix_world @ mathutils.Vector(corner) for corner in plane_obj.bound_box]
        xs = [p.x for p in plane_coords]
        ys = [p.y for p in plane_coords]
        plane_check_bbox = {
             "xmin": min(xs), "xmax": max(xs),
             "ymin": min(ys), "ymax": max(ys),
             "zmin": 0.0, "zmax": 0.0
        }

        mat = plane_obj.data.materials[0] if plane_obj.data.materials else None
        if mat:
             for site in sites:
                 # site["coordinates"] is a list [x, y, z]
                 if point_in_bbox(site["coordinates"], plane_check_bbox):
                     site["material"] = mat.name
                     site["charge"] = mat.get("charge", site.get("charge", 0.0))
                     site["potential"] = mat.get("potential", site.get("potential", 0.0))
                     site["dielectric_constant"] = mat.get("dielectric_constant", site.get("dielectric_constant", 1.0))
                     site["BCtype"] = mat.get("BCtype", site.get("BCtype", "n"))

    # Process all other objects
    print("Processing other objects...")
    count_updates = 0
    for site in sites:
        pt_list = site["coordinates"]
        pt_vec = mathutils.Vector(pt_list) # Create vector only for query
        
        for item in prepared_objects:
            if point_in_bbox(pt_list, item["bbox"]):
                if is_point_inside_object(pt_vec, item["bvh"], None):
                    # update site
                    obj = item["obj"]
                    if obj.data.materials:
                        mat = obj.data.materials[0]
                        # replicate original conditional logic
                        if (mat.name == 'dielectric' and site.get("material") == "Qsystem"):
                            pass
                        else:
                            site["material"] = mat.name
                            site["charge"] = mat.get("charge", site.get("charge", 0.0))
                            site["potential"] = mat.get("potential", site.get("potential", 0.0))
                            site["dielectric_constant"] = mat.get("dielectric_constant", site.get("dielectric_constant", 1.0))
                            site["BCtype"] = mat.get("BCtype", site.get("BCtype", "n"))
                    # optimization: break? Original script checks all objects but has a break?
                    # Original: "if is_point_inside... break"
                    break
        
    print(f"Exporting to {outfile}...")
    with open(outfile, "w") as f:
        json.dump(sites, f, indent=4)
        
    print(f"Done in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
