bl_info = {
    "name" : "Geometrical heart reconstrucion", 
    "author" : "Daniel Verhuelsdonk",
    "version" : (1, 23),
    "blender" : (3, 1, 0),
    "location" : "Operator Search",
    "description": "Panel and operators to geometrically reconstruct the upper heart shape",
    "warning" : "",
    "wiki_url" : "",
    "category" : "Add Mesh",
}
# Imports
import bpy
import bmesh
import math
import mathutils 
import numpy as np
import open3d as o3d
scene = bpy.types.Scene

# Generally used functions.
def cons_print(data):
    """Print to console for button presses. Used for error messages, information outputs and warnings."""
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'CONSOLE':
                override = {'window': window, 'screen': screen, 'area': area}
                bpy.ops.console.scrollback_append(override, text=str(data), type="OUTPUT")   

def copy_object(input_name, output_name):
    """Copy object with given name."""
    src_obj = bpy.data.objects[input_name]
    new_obj = src_obj.copy()
    new_obj.data = src_obj.data.copy()
    new_obj.data.name = output_name 
    new_obj.animation_data_clear()
    new_obj.name = output_name
    bpy.context.collection.objects.link(new_obj)
    return new_obj

def deselect_object_vertices(obj):
    """Go into edit mode and deselect all vertices of an object."""
    # Transfer data into edit mode.
    me = obj.data
    bpy.ops.object.mode_set(mode='EDIT') 
    bm = bmesh.from_edit_mesh(me)
    # Deselect all vertices.
    for v in bm.verts: v.select = False
    # Return to object mode and update the mesh to the obeject.
    bm.select_flush_mode()   
    me.update()
    bpy.ops.object.mode_set(mode='OBJECT') 

def join_objects(obj, joined_obj):
    """Join two objects without changing the selection."""
    # Set correct selections before joining geometries.
    bpy.ops.object.mode_set(mode='OBJECT') 
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    joined_obj.select_set(True)
    bpy.ops.object.mode_set(mode='OBJECT') 
    bpy.ops.object.join() # Join objects.

def get_selected_edges(obj):
    """Return currenctly selected edges and corresponding vertices of active object."""
    mesh = bmesh.from_edit_mesh(obj.data)
    active_edges_verts, active_edges_indices = ([] for i in range(2))
    for e in mesh.edges:
        if e.select:
            active_edges_verts.append((e.verts[0].index, e.verts[1].index))
            active_edges_indices.append(e.index)
    return active_edges_indices, active_edges_verts

def list_diff(list1, list2):
    """Difference between two lists."""
    li_dif = [i for i in list1 + list2 if i not in list1 or i not in list2]
    return li_dif

def transfer_data_to_mesh(obj):
    """Transfer input object data to mesh and return bmesh object."""
    bm = bmesh.new()       
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    return bm

# UI-interface functions.
class MESH_OT_get_node(bpy.types.Operator):
    """Get node position coordinates and save coordinates in UI."""
    bl_idname = 'heart.get_point'
    bl_label = 'Get node position coordinates and save coordinates in UI.' 
    point_mode: bpy.props.StringProperty(name = "point_mode", description="Which point to select", default = "Top") # Which point is selected with this function(Top/basal, bottom/apical, septum)
    def execute(self, context):
        obj = context.object
        if obj.mode != 'EDIT': # Exception when in object mode.
            cons_print('This process only works in edit mode.')
            return{'CANCELLED'} 
        # Get vector-coordinates.
        bm = bmesh.from_edit_mesh(obj.data)
        counter = 0
        for v in bm.verts:
            if v.select:
                counter += 1 # Increment to get the number of selected nodes
                vertice_coords = obj.matrix_world @ v.co # Transform to global coordinate system
                index = v.index
        # Check for the correct number of nodes selected.
        if counter !=1:
            cons_print('Incorrect amount of nodes. Only one node may be selected.')
            return{'CANCELLED'}  
        # Using temporary veriables prevents the change of the global variable, when to many nodes are selected.
        if self.point_mode == "Top":  
            context.scene.pos_top = vertice_coords
            context.scene.top_index = index # Update index of the top position.   # !!! kann raus sobald remove_basal neu ist 
        elif self.point_mode == "Bot": context.scene.pos_bot = vertice_coords
        elif self.point_mode == "Septum": context.scene.pos_septum = vertice_coords
        else:
            cons_print('Unsupported point mode input. Only Top, Bot and Septum available.')
            return{'CANCELLED'}    
        return{'FINISHED'}

class MESH_OT_ventricle_rotate(bpy.types.Operator):
    """Rotate Ventricle using the node-coordinates of the basal, apical and septum node."""
    bl_idname = 'heart.ventricle_rotate'
    bl_label = 'Rotate Ventricle using the node-coordinates of the basal, apical and septum node.' 
    def execute(self, context): 
        if not rotate_ventricle(context): return{'CANCELLED'}
        return{'FINISHED'}

def rotate_ventricle(context):
    """Rotate ventricle geometry using three points on the ventricle."""
## Coniditions to terminate the code.
    if bpy.context.mode != 'OBJECT': bpy.ops.object.editmode_toggle() # Toggle to object mode.
    if len(bpy.context.selected_objects) < 1:# Only works if and object is selected
        cons_print("No object selected.")
        return False
## Precompute the rotation angles using the relative positions between top, bottom and septum node.
    # Initialize points.
    top = mathutils.Vector((context.scene.pos_top[0], context.scene.pos_top[1], context.scene.pos_top[2]))
    bottom = mathutils.Vector((context.scene.pos_bot[0], context.scene.pos_bot[1], context.scene.pos_bot[2]))
    septum = mathutils.Vector((context.scene.pos_septum[0], context.scene.pos_septum[1], context.scene.pos_septum[2]))
    vec_difference = np.array([top.x - bottom.x, top.y - bottom.y, top.z - bottom.z]) # Compute difference between Top and Bottom to compute the angle for the first rotation.
    # First rotation precomputation - X.
    angle_x = get_rotation_angle(vec_difference[1], vec_difference[2]) # Get angle.
    # Compute rotated top and septum node using rotation matrix around x-axis.
    rot_matrix = np.array([[1, 0, 0], [0, math.cos(angle_x), -math.sin(angle_x)], [0, math.sin(angle_x), math.cos(angle_x)]]) 
    rot_top = rot_matrix.dot(vec_difference) 
    rot_septum = rot_matrix.dot(septum-bottom) 
    # Second rotation precomputation - Y.
    angle_y = - get_rotation_angle(rot_top[0], rot_top[2]) # Get angle. Turn the rotation direction with a minus sign.
    # Compute second rotation of septum node using rotation matrix around y-axis.
    rot_matrix_two = np.array([[math.cos(angle_y), 0, math.sin(angle_y)], [0, 1, 0], [-math.sin(angle_y), 0, math.cos(angle_y)]]) 
    double_rot_top = rot_matrix_two.dot(rot_top)
    double_rot_septum = rot_matrix_two.dot(rot_septum)
    context.scene.pos_top = (round(abs(double_rot_top[0]), 6), round(abs(double_rot_top[1]), 6), round(abs(double_rot_top[2]), 6)) # Update UI top-variables.
    # Third rotation precomputation - Z.
    angle_z = get_rotation_angle(double_rot_septum[0], double_rot_septum[1]) # Get angle.
    # Compute third rotation of septum node using rotation matrix around y-axis.
    rot_matrix_three = np.array([[math.cos(angle_z), -math.sin(angle_z), 0], [math.sin(angle_z), math.cos(angle_z), 0], [0, 0, 1]]) 
    third_rot_septum = rot_matrix_three.dot(double_rot_septum)
    context.scene.pos_septum = (round(abs(third_rot_septum[0]), 6), round(abs(third_rot_septum[1]), 6), round(abs(third_rot_septum[2]), 6)) # Update UI septum-variables.
## Translation and rotation-process for all selected objects.
    # Translation.
    if context.scene.pos_bot != (0, 0, 0):   
        # Translate Coordinate system to (0,0,0) by subtracting the bottom node for easier rotation
        bpy.ops.transform.translate(value=(-bottom), orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        bpy.ops.object.transform_apply(location=True, scale=False, rotation=False)
        # Update UI bottom-variables.
        context.scene.pos_bot = (0, 0, 0)
    # Rotation operations around x-, y- and z-axis. 
    bpy.ops.transform.rotate(value=angle_x, orient_axis='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
    bpy.ops.transform.rotate(value=angle_y, orient_axis='Y', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
    bpy.ops.transform.rotate(value=angle_z, orient_axis='Z', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
    # Apply all transformations.
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return True

def get_rotation_angle(numerator, denominator):
    """Function to quickly and reliably compute the rotation angle. Python tan-functions did not give the correct angles for some cases."""
    angle = 0
    if denominator > 0: angle = math.atan(numerator/denominator)
    elif denominator < 0:
        if numerator > 0: angle = math.pi + math.atan(numerator/denominator)
        elif numerator < 0: angle = - math.pi + math.atan(numerator/denominator)
    else:
        if numerator > 0: angle = math.pi    
        elif numerator < 0: angle = - math.pi 
    return angle

class MESH_OT_cut_edge_loops(bpy.types.Operator):
    """Remove the basal region of the ventricle by deleting edge loops of the ventricle from the pre-selected top position."""
    bl_idname = 'heart.cut_edge_loops'
    bl_label = 'Remove the basal region of the ventricle by deleting edge loops of the ventricle from the pre-selected top position.'
    def execute(self, context): 
        selected_objects = context.selected_objects
        cut_edge_loops(context, selected_objects)
        for obj in selected_objects: obj.select_set(True) # Reselect objects.
        return{'FINISHED'}  

def cut_edge_loops(context, selected_objects):
    """Function to remove the upper edge loops of the largest ventricle."""
    for obj in selected_objects: obj.select_set(False) # Initialize objects deselected.
    for obj in selected_objects:
        # Select object,set is as active and deselect all its vertices.
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        deselect_object_vertices(obj)
        # Cut basal region off of the ventricle.
        dissolve_edge_loops(context, obj)
        obj.select_set(False) # Deselect object, after removing basal region.

def get_neighbour_vertices(v): # !!! can maybe be shortend to one line.
    """Return neighbouring vertices of a vertex."""
    neighbours_index = []
    for e in v.link_edges: neighbours_index.append(e.other_vert(v).index)
    return neighbours_index

def dissolve_edge_loops(context, obj): 
    """Dissolve a given amount of edge loops."""
    bpy.context.tool_settings.mesh_select_mode = (True, False, False) # Make sure vertex mode is selected in edit mode.
    bm = transfer_data_to_mesh(obj) # Transfer object in mesh data.
    # Initialize neighbour vertices list with a single vertex.
    inner_loop = [] # Inner loop.
    inner_loop.append(context.scene.top_index) # Inner loop starts with the top position vertex.
    # Dissolve an amount of edge loops around a selected point.
    for i in range(context.scene.amount_of_cuts + 1):
        outer_loop = [] # Neighbouring loop of the inner loop.
        # Find vertex in top-positon.
        for v in bm.verts:
            if v.index in inner_loop:
                v.select = True  # Select vertices to dissolve.
                neighbours_index = get_neighbour_vertices(v) 
                for n in neighbours_index:# Get neighbours of the current vertex in the inner loop.
                    if n not in outer_loop: outer_loop.append(n)     
            else: v.select = False
        # Update outer loop in inner loop for the next timestep.
        inner_loop = outer_loop
    bm.to_mesh(obj.data) # Update geometry after deleting face.
    # Dissolve faces.
    bpy.ops.object.mode_set(mode='EDIT')  
    bpy.ops.mesh.dissolve_faces()
    bpy.ops.object.mode_set(mode='OBJECT') 
    # Load mesh data from current object for second operation.
    bm_two = bmesh.new()       
    bm_two.from_mesh(obj.data)
    bm_two.faces.ensure_lookup_table()
    # Get currently selected verts. This should be the vertices of the last loop .
    selected_verts = [v.index for v in bm_two.verts if v.select]
    # Delete last remaining face between last loop.
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.delete(type='FACE')
    bpy.ops.object.mode_set(mode='OBJECT') 
    # Reselect vertices of edge loop after face removal.
    for v in obj.data.vertices:
        if v.index in selected_verts: v.select = True
        else: v.select = False

    # Save upper apical edge loop in a vertex group to be selected again later on.
    bm = transfer_data_to_mesh(obj)
    marked_verts = [v.index for v in bm.verts if v.select]
    vg_upper_apical = "upper_apical_edge_loop"
    vg_orifice = obj.vertex_groups.new(name = vg_upper_apical)
    vg_orifice.add(marked_verts, 1, 'ADD' )

    subdivide_last_edge_loop(obj, vg_orifice) # Apply subdivide to smooth out further connection.

class MESH_OT_new_remove_basal(bpy.types.Operator): 
    """Remove the basal region using a threshold value."""
    bl_idname = 'heart.remove_basal'
    bl_label = 'Remove basal region.'
    def execute(self, context): 
        remove_multiple_basal_region(context)
        return{'FINISHED'} 

def remove_multiple_basal_region(context):
    """Remove multiple basal regions selecting the EDV as the reference element."""
    cons_print(f"Removing original basal region.")
    # Read selected objects.
    if not context.selected_objects:
        cons_print("No elements selected.")
        return False
    selected_objects = context.selected_objects
    # Find object with max volume and create a copy of it as a reference object.
    reference_copy = copy_object(find_reference_ventricle_name(selected_objects), 'basal_region')
    # Deselect objects.
    for obj in selected_objects: obj.select_set(False)

    deleted_verts = remove_basal_region(context, reference_copy, []) # Remove from reference object
    cons_print(f"Deleted verts: {deleted_verts}")

    for obj in selected_objects: remove_basal_region(context, obj, deleted_verts) # Remove from reference object (mean volume should be best)!!!.
    # Compute volumes
    # Remove for reference
    # remove for rest

def remove_basal_region(context, obj, del_nodes): #!!!
    """Remove basal region of the ventricle using a threshold."""
    if obj.mode == 'EDIT': bpy.ops.object.mode_set(mode='OBJECT') # Toggle to object mode.
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    deselect_object_vertices(obj)
    
## Select vertices to delete (above z-coordinate threshold).
    bm = transfer_data_to_mesh(obj) 
    cons_print(f"Length bm.verts: {len(bm.verts)}")
    if del_nodes == []: # Initialisation of deleted node selection for reference object.
        for v in bm.verts:
            vertice_coords = obj.matrix_world @ v.co # Transfer to global coordinate system.
            
            if vertice_coords[2] > context.scene.remove_basal_threshold: # Only vertices above threshold are selected to be deleted.
                v.select = True
                del_nodes.append(v.index)
                cons_print(f"Vertex: {v.index} at ({round(vertice_coords[0],2)}|{round(vertice_coords[1],2)}|{round(vertice_coords[2],2)})")
    else:
        for v in bm.verts:
            if v.index in del_nodes: v.select = True
    bm.to_mesh(obj.data) # Update selection to object.
    ## Remove selected nodes.
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.delete_edgeloop()
    bpy.ops.object.mode_set(mode='OBJECT')
## Save upper apical edge loop in a vertex group to be selected again later on.
    bm = transfer_data_to_mesh(obj)
    marked_verts = [v.index for v in bm.verts if v.select]
    vg_upper_apical = "upper_apical_edge_loop"
    vg_orifice = obj.vertex_groups.new(name = vg_upper_apical)
    vg_orifice.add(marked_verts, 1, 'ADD' )
## Remove remaining face create after delete_edgeloop().
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.delete(type='FACE') 
## Refinement
    vg_orifice = refine_upper_apical_edge_loop(obj, vg_orifice)
    smooth_apical_region(context, obj, vg_orifice)
    return del_nodes #!!! apply to other ventricles. currently only reference, to do fuer die anderen ventricle

def refine_upper_apical_edge_loop(obj, vg_orifice):
    """!!!"""
    # Select upper apical edge loop with vertex group.
    bpy.ops.object.vertex_group_set_active(group=vg_orifice.name)
    bpy.ops.object.vertex_group_select()
    # Smooth highest edge loop of apical region aligning the vertices onto a plane. !!! hat noch verbesserungsbedarf. Alle punkte auf einheitliche Hoehe waere gut.
    bpy.ops.mesh.looptools_relax(input='selected', interpolation='linear', iterations='5', regular=True) # Reduce spikes on the highest edge loop.
    bpy.ops.mesh.looptools_flatten(influence=100, lock_x=False, lock_y=False, lock_z=False, plane='best_fit', restriction='none') # Flatten highest edge loop onto a plane
    # Subdivide last edge loop   
    # !!! get amount of upper apical and lower basal edge loops to optimize this subdivision
    return subdivide_last_edge_loop(obj, vg_orifice)
    
def subdivide_last_edge_loop(obj, vg_orifice): #!!! maybe erweitern durch anzahl subdivisions in abhaengigkeit des verhaeltnisses zwischen der anzahl der oberen apikalen und unteren basalen nodes.
    """Subdivide last edge loop in two steps before bridging for a better transition between coarse apical and fine basal mesh."""
    # Select upper apical edge loop with vertex group.
    deselect_object_vertices(obj)
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.object.vertex_group_set_active(group=vg_orifice.name)
    bpy.ops.object.vertex_group_select()
    # Iteratively subdivide edge loop with surrounding mesh creating a less skewed transition between subdivides and unsibdivided elements.
    bpy.ops.mesh.select_more()
    bpy.ops.mesh.subdivide(number_cuts=1, ngon=False)
    bpy.ops.mesh.select_less()
    bpy.ops.mesh.select_less()
    bpy.ops.object.mode_set(mode='OBJECT')
    # Re-initialize vertex group for upper apical edge loop.
    bpy.ops.object.mode_set(mode='OBJECT')
    bm = transfer_data_to_mesh(obj)
    selected_verts = [v.index for v in bm.verts if v.select] # Re-read selected vertices.
    vg_name = vg_orifice.name
    if vg_orifice is not None: obj.vertex_groups.remove(vg_orifice)
    vg_orifice = obj.vertex_groups.new(name = vg_name)
    vg_orifice.add(selected_verts, 1, 'ADD' )
    return vg_orifice

def smooth_apical_region(context, obj, vg_orifice):
    """Smooth apical region  in the region of the cut.."""
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.vertex_group_set_active(group=vg_orifice.name)
    deselect_object_vertices(obj)
    # Find vertices to select initially for smoothing.
    bpy.ops.object.mode_set(mode='OBJECT') 
    bm = transfer_data_to_mesh(obj)
    for v in bm.verts: # Select vertices by z-threshold.
        vertice_coords = obj.matrix_world @ v.co 
        if vertice_coords[2] > 9 / 10 * context.scene.remove_basal_threshold: v.select = True
    bm.to_mesh(obj.data) # Transfer selection to object.
    bpy.ops.object.mode_set(mode='EDIT') 
    # First strong smoothing operation.
    bpy.ops.object.vertex_group_deselect()
    bpy.ops.mesh.vertices_smooth(factor=0.4, repeat=10)
    n_smooth_iter = 3
    for i in range(n_smooth_iter): # Continuously weaker smoothing operations with more nodes to create a smooth transition between smoothed and unsmoothed region.
        bpy.ops.mesh.select_more()
        bpy.ops.object.vertex_group_deselect()
        bpy.ops.mesh.vertices_smooth(factor=0.4, repeat=3-i)
    bpy.ops.object.mode_set(mode='OBJECT')

class MESH_OT_build_valve(bpy.types.Operator):
    """Create geometry for mitral and aortic valve."""
    bl_idname = 'heart.build_valve'
    bl_label = 'Create geometry for mitral or aortic valve.'
    def execute(self, context):
        ratio_annulli = 1
        for obj in context.selected_objects:
            if not build_both_valves(context, obj, ratio_annulli): return{'CANCELLED'} 
            merge_overlap(threshold = 0.0001) # Protection against double insertion of valves.
        return{'FINISHED'}

def build_both_valves(context, obj, ratio):
    """Build both the aortic and mitral valve."""
    aortic_min = build_valve(context, obj, valve_mode = "Aortic" , ratio = ratio)
    mitral_min = build_valve(context, obj,  valve_mode = "Mitral", ratio =  ratio)
    return aortic_min, mitral_min

def build_valve(context, obj,  valve_mode, ratio):
    """Build ventricle valve and connect it to current geometry."""
    if valve_mode == "Aortic": obj_name = "por_Boundary_AV"
    elif valve_mode == "Mitral": 
        if context.scene.bool_porous: obj_name = "por_Boundary_MV"
        else: obj_name = "Boundary_MV"
    else: return False
    return add_and_join_object(context, obj, obj_name, valve_mode, ratio)

def add_and_join_object(context, obj, new_obj_name, valve_mode, ratio):
    """Add an object and join it with the currently selected geometries."""
    # Create new object by copying it from existing object.
    new_obj = copy_object(new_obj_name, "new_obj")
    new_obj.select_set(True)
    bpy.context.view_layer.objects.active = new_obj
    # Modify the copied object to fit in the correct place.
    scale_rotate_translate_object(context, new_obj, valve_mode, ratio)
    maxim, minim = get_min_max(new_obj)
    # Combine new object with selected object.
    join_objects(obj, new_obj) 
    return minim[2]

def scale_rotate_translate_object(context, obj, valve_mode, ratio):
    """Scale, rotate and shift object and save transformation."""
    translation, angles, radius_vertical, radius_horizontal = get_valve_data(context, valve_mode) # Get scale from UI-data.
    obj.scale = (ratio * radius_vertical, ratio * radius_horizontal, ratio * (radius_horizontal + radius_vertical) / 2) # Scale object with given ratio.
    obj.rotation_euler = angles # Rotate object by input angles.
    obj.location = translation # Translate object with input translation.
    if obj.mode == 'OBJECT': bpy.ops.object.mode_set()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True) # Apply changes.
    bpy.ops.object.mode_set()

def get_min_max(obj):
    """Return smallest and highest value of an object in each dimension."""
    maxim = np.array([-1000.0, -1000.0, -1000.0])
    minim = np.array([1000.0, 1000.0, 1000.0])
    for p in obj.data.vertices: # Find maxima and minima.
        vertice_coords = obj.matrix_world @ p.co # Transform to global coordinate system.
        if vertice_coords[0] > maxim[0]: maxim[0] = vertice_coords[0]
        if vertice_coords[0] < minim[0]: minim[0] = vertice_coords[0]
        if vertice_coords[1] > maxim[1]: maxim[1] = vertice_coords[1]
        if vertice_coords[1] < minim[1]: minim[1] = vertice_coords[1]
        if vertice_coords[2] > maxim[2]: maxim[2] = vertice_coords[2]
        if vertice_coords[2] < minim[2]: minim[2] = vertice_coords[2]
    return maxim, minim

def merge_overlap(threshold):
    """Merge overlapping vertices of currently selected object."""
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=threshold, use_sharp_edge_from_normals=False, use_unselected=False)
    bpy.ops.object.mode_set(mode='OBJECT') 

class MESH_OT_support_struct(bpy.types.Operator):
    """Create supporting structure to help poisson algorithm accurately build basal ventricle region."""
    bl_idname = 'heart.support_struct'
    bl_label = 'Create supporting structure to help poisson algorithm accurately build basal ventricle region.'
    def execute(self, context):
        for obj in context.selected_objects:
            build_support_structure(context, obj, ratio=1.1)
            merge_overlap(threshold = 0.0001) # Protection against double insertion of valves.
        return{'FINISHED'}

def build_support_structure(context, obj, ratio):
    """Build support structure to help the poisson surface reconstrucion algorithm create a smooth surface after Poisson surface reconstruction."""
    aortic_min_up, mitral_min_up = build_both_valves(context, obj, ratio) # Larger annulus structure (upscaled).
    aortic_min_down, mitral_min_down =  build_both_valves(context, obj, 1 / ratio) # Smaller annulus structure (downscaled).
    return aortic_min_up, mitral_min_up, aortic_min_down, mitral_min_down

class MESH_OT_poisson(bpy.types.Operator):
    """Apply Poisson surface reconstrucion to point cloud creating a surface mesh."""
    bl_idname = 'heart.poisson'
    bl_label = 'Apply Poisson surface reconstrucion to point cloud creating a surface mesh.'
    def execute(self, context):
        if len(bpy.context.selected_objects) == 0 or bpy.context.mode != 'OBJECT': return{'CANCELLED'} # Check if any object is selected and correct context mode.
        for object in bpy.context.selected_objects: create_poisson_from_object_pointcloud(context, object) # Repeat Poisson-surface-reconstruction algorithm for all selected objects.
        return{'FINISHED'}

def create_poisson_from_object_pointcloud(context, obj):
    """Create poisson surface reconstruction for a single point cloud."""
    point_data = np.asarray(obj.data.vertices) # Get point data of current object.
    # Initialize and fill entries of object vertices.
    object_vertices = np.empty(shape=[0, 3])
    for point in point_data:
        vertice_coords = obj.matrix_world @ point.co # Rotate points to put them in global coordinate system.
        # Append new vertex to point cloud array.
        new_point = np.array([[vertice_coords[0], vertice_coords[1], vertice_coords[2]]])
        object_vertices = np.concatenate((object_vertices, new_point), axis=0)
    # Create Point cloud object and fill it with points.
    point_cloud_data = o3d.geometry.PointCloud()
    point_cloud_data.points = o3d.utility.Vector3dVector(object_vertices)
    # Prepare point cloud for poisson surface reconstruction. It needs the normals of the points in the point cloud.
    point_cloud_data.estimate_normals()
    point_cloud_data.normalize_normals()
    point_cloud_data.orient_normals_consistent_tangent_plane(20)
    # Apply poisson surface reconstruction.
    poisson_mesh, poisson_dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud_data, depth = context.scene.poisson_depth, width=0, scale=1.1, linear_fit=False, n_threads= 1)
    # Initialize empty arrays for object data and assign faces vertices and edges.
    vertices, edges, faces = ([] for i in range(3))
    vertices = poisson_mesh.vertices
    faces = poisson_mesh.triangles
    # Create object from vertices, edges and faces in Blender.
    emptyMesh = bpy.data.meshes.new('emptyMesh')
    emptyMesh.from_pydata(vertices, edges, faces)
    emptyMesh.update()
    poisson_obj = bpy.data.objects.new(obj.name + "_poisson", emptyMesh)    
    bpy.context.collection.objects.link(poisson_obj)
    # Change selection.
    obj.select_set(False)
    poisson_obj.select_set(True)
    bpy.context.view_layer.objects.active = poisson_obj
    obj.hide_set(True)
    return poisson_obj

class MESH_OT_create_valve_orifice(bpy.types.Operator):
    """Remove vertices blocking the valve entries into a single face, delete this face and create a vertex group for all vertices around that face."""
    bl_idname = 'heart.create_valve_orifice'
    bl_label = 'Remove vertices blocking the valve entries into a single face, delete this face and create a vertex group for all vertices around that face.'
    def execute(self, context):
        # Check for the right amount of objects and context mode.
        if len(bpy.context.selected_objects) == 0 or bpy.context.mode != 'OBJECT': return{'CANCELLED'}
        for obj in context.selected_objects:
            deselect_object_vertices(obj) # Deselect all vertices as preparation for the deletion of the valve orifice.
            # Delete nodes in areas around the valves to create valve orifices.
            create_valve_orifice(context, "Aortic")
            create_valve_orifice(context, "Mitral")
        return{'FINISHED'}

def create_valve_orifice(context, valve_mode): 
    """Create orifice in the geometry, where valve interface nodes will be inserted later."""
    # Dissolve vertices inside of a given area-object placed around the valve vertices.
    select_valve_vertices(context, valve_mode) 
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.dissolve_verts()
    bpy.ops.object.mode_set(mode='OBJECT')
    # Remove face remaining after dissolving vertices.
    translation, angles, radius_vertical, radius_horizontal = get_valve_data(context, valve_mode)
    obj = bpy.context.active_object
    if obj.mode == 'EDIT': bpy.ops.object.mode_set() # Toggle to object mode.
    bpy.context.tool_settings.mesh_select_mode = (False, True, False)
    bm = transfer_data_to_mesh(obj)
    vertices_orifice = []
    # Select vertices of newly created face after dissolving.
    for f in bm.faces:
        f.select = False
        # Compute difference between center and translation vector and check if it is smaller than the smaller valve radius.
        if distance_vec(f.calc_center_median(), translation) < min(radius_vertical, radius_horizontal)  / 2: 
            for v in f.verts: vertices_orifice.append(v.index)
            f.select = True
    # Delete face in orifice.
    faces = [f for f in bm.faces if f.select]
    bmesh.ops.delete(bm, geom = faces, context = 'FACES_ONLY')
    bm.to_mesh(obj.data)
    # Create vertex group containing orifice edge loop vertices.
    vg_orifice = obj.vertex_groups.new( name = f"{valve_mode}_orifice")
    vg_orifice.add( vertices_orifice, 1, 'ADD')
    # Remove troubling vertices(vertices with 2 neighbours) in (currently selected) orifice vertex group and smooth this edge loop.
    smooth_relax_edgeloop(obj, vg_orifice) 
    """ !!! funktioniert bei Janas geometrien auch ohne, weil die so groß sind. Vielleicht muss man das mit klappen skalieren.
    # Subdivide for the real mitral valve for a smoother transition.
    if valve_mode == "Mitral" and context.scene.bool_porous: 
        #!!!select more #Smoother mesh transition, subdivide last edge loop
        bpy.ops.mesh.subdivide(number_cuts=1, ngon=False)
        # select less
        # subdivide
        bpy.ops.object.mode_set(mode='OBJECT')
        vg_orifice.add( vertices_orifice, 1, 'ADD')
        bpy.ops.object.mode_set(mode='EDIT')"""
    bpy.ops.object.mode_set(mode='OBJECT')
    return True

def select_valve_vertices(context, valve_mode):
    """Select all vertices of a given valve."""
    if 'Valve_area' in [obj.name for obj in bpy.data.objects]: bpy.data.objects.remove(bpy.data.objects['Valve_area'], do_unlink=True) # Remove valve area object, if an object with the same name already exists.
    original_obj = bpy.context.active_object
    if valve_mode == "Mitral":
        if context.scene.bool_porous: copy_object('por_Area_MV', 'Valve_area')
        else: copy_object('Area_MV', 'Valve_area')
    elif valve_mode == "Aortic": copy_object('Area_AV', 'Valve_area')
    else: return False
    valve_area = bpy.data.objects['Valve_area']
    # Rescale, translate and rotate valve area.
    scale_rotate_translate_object(context, valve_area, valve_mode, ratio=1.025) # !!! scale of valve_area should be good, but maybe a context UI variable might be better.
    # Select all vertices in the valve area object.
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action="DESELECT")
    mesh=bmesh.from_edit_mesh(bpy.context.object.data)
    cut_obj_matrix = valve_area.matrix_world.inverted()
    mat = cut_obj_matrix @ original_obj.matrix_world         
    selected_verts = [v
        for v in mesh.verts
        if is_inside((mat @ v.co), valve_area)]
    for v in selected_verts: 
        if v.co.z > -1: v.select = True
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.data.objects.remove(bpy.data.objects[valve_area.name], do_unlink=True) # Remove valve area object.
    
def distance_vec(point1, point2) -> float:
    """Calculate distance between two nodes."""
    return math.sqrt((point2[0] - point1[0]) ** 2 +
                     (point2[1] - point1[1]) ** 2 +
                     (point2[2] - point1[2]) ** 2)

def is_inside(p, cut_obj):
    """Check if a point p is inside the object cut_obj."""
    result, point, normal, face = cut_obj.closest_point_on_mesh(p)
    if not result: return False
    p2 = point-p
    v = p2.dot(normal)
    return not (v < 0.0)

def get_valve_data(context, valve_mode):
    """ Return data for specific valve type"""
    if valve_mode == "Aortic":
        translation = context.scene.translation_aortic
        angles = np.radians(context.scene.angle_aortic)
        radius_vertical = context.scene.aortic_radius
        radius_horizontal = context.scene.aortic_radius
    elif valve_mode == "Mitral":
        translation = context.scene.translation_mitral
        angles = np.radians(context.scene.angle_mitral)
        radius_vertical = context.scene.mitral_radius_long
        radius_horizontal = context.scene.mitral_radius_small
    else: return False, False, False, False
    return translation, angles, radius_vertical, radius_horizontal 
    
class MESH_OT_connect_valves(bpy.types.Operator):
    """Connect valve interface nodes with orifice for mitral and aortic valve."""
    bl_idname = 'heart.connect_valve'
    bl_label = 'Connect valve interface nodes with orifice for mitral and aortic valve.'
    def execute(self, context):
        connect_valve_orifice(context, "Aortic", valve_index = 4)
        connect_valve_orifice(context, "Mitral", valve_index = 4)
        return{'FINISHED'}

def connect_valve_orifice(context, valve_mode, valve_index):
    """Connect orifices around valves with surrounding mesh nodes."""
    obj = bpy.context.active_object
    build_valve_surface(context, obj, valve_mode = valve_mode, ratio = 1, valve_index = valve_index) # Create interface valve nodes.
    bpy.ops.object.mode_set(mode='EDIT') # Change selection mode in edit mode for brige loop operator.
    bpy.ops.mesh.bridge_edge_loops()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)

def smooth_relax_edgeloop(obj, vg_orifice):
    """Relax selected edge loop such that vertices with only two edges get deleted before relaxation, because these vertices will create skew triangles, when using the bridge_edgeloop() command."""
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.context.tool_settings.mesh_select_mode = (True, False, False) # Activate edge mode in edit mode.
    i = 0 # Breakup-condition for while-loop to prevent infinite loops.
    while select_vertices_outside_of_edge_loop(obj) and i < 100: # Select vertices with only two neighbours.
        i = i + 1
        bpy.ops.object.mode_set(mode='OBJECT') # Update Mehs in Blender.
        bpy.ops.object.mode_set(mode='EDIT') 
        bpy.ops.mesh.delete(type='VERT') # Remove selected vertices.
        # Reselect vertices in orifice vertex group.
        bpy.ops.object.vertex_group_set_active(group=str(vg_orifice.name))
        bpy.ops.object.vertex_group_select()
    # Select vertices in vertex group of orifice vertices.
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.object.vertex_group_set_active(group=str(vg_orifice.name))
    bpy.ops.object.vertex_group_select()
    # Relax orifice loop to reduce spikes in transition to valve.
    bpy.ops.mesh.looptools_relax(input='selected', interpolation='linear', iterations='5', regular=True)
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)

def select_vertices_outside_of_edge_loop(obj):
    """Select vertices in an edge loop with two or less neighbours."""
    # Transfer data into edit mode.
    me = obj.data
    bpy.ops.object.mode_set(mode='EDIT') 
    bm = bmesh.from_edit_mesh(me)
    # Initialize return type
    must_remove = False
    # Select vertices in edge loop with only two connecting vertices.
    for v in bm.verts:
        if len(get_neighbour_vertices(v)) <= 2 and v.select: #v.index in vertices_orifice: 
            v.select = True
            must_remove = True
        else: v.select = False
    # Return to object mode and update the mesh to the obeject.
    bm.select_flush_mode()   
    me.update()
    bpy.ops.object.mode_set(mode='OBJECT') 
    return must_remove

def build_valve_surface(context, obj, valve_mode, ratio, valve_index):
    """Copy valve object(mitral/aortic), combine it with the active geometry and select only boundary vertices. Boundary_only allows to delete all vertices except the boundary edge loop on the ventricle side."""
    # Assign names for further operations:
    if valve_mode == "Aortic": 
        obj_name = "por_AV_surf"
        vg_boundary = "AV_Boundary"   # Vertex group of the boundary of the valve depending on valve_mode (MV_Boundary/AV_Boundary):
        vg_orifice = "Aortic_orifice"
    elif valve_mode == "Mitral": 
        vg_boundary = "MV_Boundary"
        vg_orifice = "Mitral_orifice"
        if context.scene.bool_porous: 
            obj_name = "por_MV_surf"
        else:
            if valve_index in range(5):
                obj_name = f"MV_real_{valve_index}"
            else:
                return False
    else: return False
    add_and_join_object(context, obj, obj_name, valve_mode, ratio) # Add valve object for the given valve type.
    # Select valve boundary and orifice vertices.
    deselect_object_vertices(obj)
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.object.vertex_group_set_active(group=str(vg_boundary))
    bpy.ops.object.vertex_group_select()
    bpy.ops.object.vertex_group_set_active(group=str(vg_orifice))
    bpy.ops.object.vertex_group_select()
    bpy.ops.object.mode_set(mode='OBJECT') 

class MESH_OT_Add_Atrium(bpy.types.Operator):
    """Add atrium to current geometries."""
    bl_idname = 'heart.add_atrium'
    bl_label = 'Add atrium to current geometries.'
    def execute(self, context):
        add_atrium(context)
        return{'FINISHED'}

def add_atrium(context):
    """Copy atrium and place it above the mitral valve as a separate object."""
    # Choose the atrium fitting the current case.
    if context.scene.bool_porous: atrium = copy_object("por_Atrium", "atrium")
    else: atrium = copy_object("Atrium", "atrium")
    atrium.select_set(True)
    bpy.context.view_layer.objects.active = atrium
    scale_rotate_translate_object(context, atrium, "Mitral", ratio=1)
    return atrium

class MESH_OT_Add_Aorta(bpy.types.Operator):
    """Add aorta to current geometries."""
    bl_idname = 'heart.add_aorta'
    bl_label = 'Add aorta to current geometries.'
    def execute(self, context):
        add_aorta(context)
        return{'FINISHED'}
    
def add_aorta(context):
    """Copy aorta and place it above the aortic valve as a separate object."""
    aorta = copy_object("por_Aorta", "aorta")
    aorta.select_set(True)
    bpy.context.view_layer.objects.active = aorta
    scale_rotate_translate_object(context, aorta, "Aortic", ratio=1)
    return aorta

class MESH_OT_Porous_zones(bpy.types.Operator):
    """Add separate porous zone objects (valves, atrium and aorta) into workspace."""
    bl_idname = 'heart.add_porous_zones'
    bl_label = 'Add separate porous zone objects (valves, atrium and aorta) into workspace.'
    def execute(self, context):
        valve_strings = ['por_AV_imperm', 'por_AV_perm', 'por_AV_res']
        create_porous_valve_zones(context, 'Aortic', valve_strings)
        if context.scene.bool_porous:
            valve_strings = ['por_MV_imperm', 'por_MV_perm', 'por_MV_res']
            create_porous_valve_zones(context, 'Mitral', valve_strings)
        return{'FINISHED'}

def create_porous_valve_zones(context, valve_mode, valve_strings):
    """Add separate porous zone objects (valves, atrium and aorta) into workspace."""
    # Create new object by copying it from existing object.
    for obj_str in valve_strings:
        new_obj = copy_object(obj_str, f"p_{obj_str}")
        new_obj.select_set(True)
        bpy.context.view_layer.objects.active = new_obj
        scale_rotate_translate_object(context, new_obj, valve_mode = valve_mode, ratio = 1)

class MESH_OT_create_basal(bpy.types.Operator):
    """Create basal region of ventricle using the position and angles of the heart valves."""
    bl_idname = 'heart.create_basal'
    bl_label = 'Create basal region of ventricle using the position and angles of the heart valves.'
    def execute(self, context):
        if not mesh_create_basal(context): return{'CANCELLED'}
        return{'FINISHED'} 

def mesh_create_basal(context):
    """Create basal region."""
    cons_print("Create basal regions for selected ventricles...")
    # Read selected objects.
    if not context.selected_objects:
        cons_print("No elements selected.")
        return False
    selected_objects = context.selected_objects
    # Find object with mean volume and create a copy of it as a reference object to create the reference basal region from.
    reference_copy = copy_object(find_reference_ventricle_name(selected_objects), 'basal_region')
    # Deselect objects.
    reference_copy.select_set(False)
    for obj in selected_objects: obj.select_set(False)
    # Operations to create basal region of the ventricle containing valve orifices.
    # Find the largest z-value in all dissolved ventricle geometries.
    find_max_value_after_dissolve(context, selected_objects)
    # Create basal region:
    basal_regions = create_basal_region_for_object(context, reference_copy)
    if not basal_regions: return False # If an error ocurred during creation of basal region, dont continue.
    # Cleanup.
    # Reselect objects to state previous to this operation and deselect (and hide for performance) created objects.
    for obj in selected_objects: obj.select_set(True)
    for basal in basal_regions: 
        basal.select_set(False)
        basal.hide_set(True)
    # Remove old basal region objects.
    if not context.scene.bool_porous: bpy.data.objects.remove(bpy.data.objects["basal_ref"], do_unlink=True)
    bpy.data.objects.remove(bpy.data.objects["basal_region"], do_unlink=True)
    bpy.data.objects.remove(bpy.data.objects["basal_region_poisson"], do_unlink=True)
    return basal_regions

def find_reference_ventricle_name(objects): 
    """Find reference object with max volume and return its name."""
    max = 0
    for obj in objects:  
        bm = transfer_data_to_mesh(obj)
        volume = bm.calc_volume(signed=True)# Compute volume and append it to the volume list.
        if volume > max: # Find ventricle with maximum volume.
            max = volume
            bpy.types.Scene.reference_object_name = obj.name
    return bpy.types.Scene.reference_object_name

def find_max_value_after_dissolve(context, objects): #!!! very inefficient currently and useless with new removal of basal region as z_max is known
    """Find the maximal z-value in all ventricle geometries after dissolving"""
    # Copy object list.
    objects_copy = []
    for counter, obj in enumerate(objects):
        copied_obj =  copy_object(obj.name, str(counter))
        objects_copy.append(copied_obj)
    # Initialize objects and their copies deselected .
    for obj in objects: obj.select_set(False)
    for obj in objects_copy: obj.select_set(False)
    context.scene.max_apical = 0 # Reset maximum apical value.
    # Find max value in z-direction.
    for obj in objects_copy:
        # Select object,set is as active and deselect all its vertices.
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        deselect_object_vertices(obj)
        # Cut basal part of the ventricle.
        dissolve_edge_loops(context, obj)
        # Find the largest z-value in the vertices of the current object.
        max_val, min_val = get_min_max(obj)
        if max_val[2] > context.scene.max_apical: 
            context.scene.max_apical = max_val[2]
        obj.select_set(False)
    # Remove list of copied objects after finding the maximum.
    for i in range(len(objects_copy)): bpy.data.objects.remove(bpy.data.objects[str(i)], do_unlink=True)

def create_basal_region_for_object(context, reference_copy):
    """Create basal part for a given ventricle."""
    for obj in context.selected_objects: obj.select_set(False) # Deselect all objects.
    # Remove basal region of reference object.
    reference_copy.select_set(True)
    bpy.context.view_layer.objects.active = reference_copy
    deselect_object_vertices(reference_copy)
    #remove_basal_region(context, reference_copy, []) # !!!
    dissolve_edge_loops(context, reference_copy) # Function to remove basal region.
    # Add valve and support structure boundary nodes.
    aortic_min, mitral_min = build_both_valves(context, reference_copy, ratio= 1) # Valves have a annuli ratio of 1.
    aortic_min_up, mitral_min_up, aortic_min_down, mitral_min_down = build_support_structure(context, reference_copy, ratio=1.1) # Support structure for the valves have a annulie ratio of 1.1 and 1/1.1.
    # Compute height plane   
    context.scene.min_valves = np.amin([aortic_min, mitral_min, aortic_min_up, mitral_min_up, aortic_min_down, mitral_min_down]) # Compute minimal valve value for the position of the cutting plane.
    if not compute_height_plane(context): return False # Compute height plane for the removal of the apical region.
    # Poisson with remeshing and triangulation.
    poisson_basal = create_poisson_from_object_pointcloud(context, reference_copy)
    voxel_size = min(context.scene.aortic_radius, context.scene.mitral_radius_long, context.scene.mitral_radius_small) / 6 # Compute voxel_size for remesh and merge dependent of the smallest valve size.
    apply_voxel_remesh(voxel_size) # Apply Remesh for better mesh quality (remove small mesh elements with high cell skewness).
    # Triangulate remesh.
    bpy.ops.object.modifier_add(type='TRIANGULATE')
    bpy.ops.object.modifier_apply(modifier="Triangulate")
    merge_vertices(voxel_size) # Merge vertices close to each other.
    # Remove apical regions.
    remove_apical_region(context, poisson_basal)
    # Remove nodes in valve areas, insert of interface nodes and smooth the basal region.
    create_valve_orifice(context, "Mitral")
    create_valve_orifice(context, "Aortic")
    basal_regions = insert_valves_into_basal(context, poisson_basal) # Create exact inputs for the valve boundaries and connect it with the remaining basal region.
    smooth_basal_region(context, voxel_size) # Smooth basal region nodes excluding valves and lower edge loop.
    return basal_regions

def compute_height_plane(context): #!!! nicht mehr noetig mit neuem removal of basal region
    """Compute height of the plane used to cut off the apical region from the basal region of the reference geometry."""
    if context.scene.min_valves <= context.scene.max_apical: # The apical region extends over the basal region.
        cons_print(f"Error: Valves ({context.scene.min_valves}) lie beneath the highest point of the ventricle({context.scene.max_apical}). Try a different setup for valve position or dissolve loops.")
        return False
    elif context.scene.min_valves < 1.025 * context.scene.max_apical: # Basal and apical region lie very close to one another. This could lead to large kinks in the geometry.
        cons_print("Info: The basal and apical region are very close to one another. The geometry may contain large kinks especially in the connection between those regions. Try a higher dissolve loop number or higher z-value for the input valves.")
    else: pass # Basal and apical region have enough distance.
    context.scene.height_plane = (context.scene.max_apical + context.scene.min_valves) / 2 # Choose z-value between lowest valve vertex and highest basal vertex.
    return True

def apply_voxel_remesh(voxel_size):
    """Apply remesh modifier with given depth to currently active object."""
    bpy.ops.object.modifier_add(type='REMESH')
    bpy.context.object.modifiers["Remesh"].mode = 'VOXEL'
    bpy.context.object.modifiers["Remesh"].voxel_size = voxel_size
    bpy.context.object.modifiers["Remesh"].adaptivity = 0
    bpy.context.object.modifiers["Remesh"].use_smooth_shade = False
    bpy.ops.object.modifier_apply(modifier="Remesh")

def merge_vertices(voxel_size):
    """Merge all vertices dependent on voxel size to eliminate vertices close to each other and thus reducing skewness in certain areas."""
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=voxel_size / 2, use_sharp_edge_from_normals=False, use_unselected=True)
    bpy.ops.object.mode_set(mode='OBJECT')

def remove_apical_region(context, obj):
    """Remove the apical ventricle region from the geometry to create solely the basal region used for all timeframes."""
    # Find vertices to delete.
    deselect_object_vertices(obj)
    bm = transfer_data_to_mesh(obj)
    for v in bm.verts:
        vertice_coords = obj.matrix_world @ v.co # Transfer to global coordinates.
        if vertice_coords[2] < context.scene.height_plane: v.select = True # Only vertices below threshold (height-plane) shall be deleted.
        else:  v.select = False
    bm.to_mesh(obj.data) # Transfer selection to object. 
    # Remove selected vertices.
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.delete_edgeloop()
    bpy.ops.object.mode_set(mode='OBJECT')
    # Save bottom edge loop to be selected again later on.
    bm = transfer_data_to_mesh(obj)
    marked_verts = [v.index for v in bm.verts if v.select] # Saved selected vertices.
    # Create vertex group for lower basal edge loop and add selected vertices to it.
    vg_lower_basal = "lower_basal_edge_loop"
    vg_orifice = obj.vertex_groups.new( name = vg_lower_basal)
    vg_orifice.add(marked_verts, 1, 'ADD' )
    # Remove remaining face created during delete_edgeloop() to create the orifice.
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.delete(type='FACE') 
    # Select vertex group.
    bpy.ops.object.vertex_group_set_active(group=str(vg_lower_basal))
    bpy.ops.object.vertex_group_select()
    # Relax and flatten lower edge loop.   
    bpy.ops.mesh.looptools_relax(input='selected', interpolation='linear', iterations='1', regular=True) # Reduce spikes on the cutting edge loop.
    bpy.ops.mesh.looptools_flatten(influence=100, lock_x=False, lock_y=False, lock_z=False, plane='best_fit', restriction='none')

def insert_valves_into_basal(context, poisson_basal): 
    """Insert valve geometry into geometry and connect it to orifice."""
    connect_valve_orifice(context, "Aortic", 0) # Insert aortic valve disk into basal region. 
    poisson_basal.select_set(False)
    basal_regions = []
    # Porous mitral valve.
    if context.scene.bool_porous: 
        # Copy basal region.
        bpy.ops.object.mode_set(mode='OBJECT')
        curr_basal = copy_object(poisson_basal.name, f"basal_0")
        curr_basal.select_set(True)
        bpy.context.view_layer.objects.active = curr_basal
        # Connect basal region to valve (Only one basal region necessary for porous medium).
        connect_valve_orifice(context, "Mitral", valve_index = 0) 
        basal_regions.append(curr_basal)
    # Interpolated mitral valve.
    else:
        # Create reference basal region to derive node-connectivity in the valve orifice region from.
        bpy.ops.object.mode_set(mode='OBJECT')
        curr_basal = copy_object(poisson_basal.name, f"basal_ref")
        curr_basal.select_set(True)
        bpy.context.view_layer.objects.active = curr_basal
        deselect_object_vertices(curr_basal)
        bpy.ops.object.mode_set(mode='EDIT')
        # Connect basal region to valve.
        edges_reference = connect_valve_orifice_reference(context, "Mitral", valve_index = 1) 
        for i in range(5): # Use reference connection for the other basal regions.
            # Copy basal region and prepare it for mitral valve insertion.
            bpy.ops.object.mode_set(mode='OBJECT')
            curr_basal = copy_object(poisson_basal.name, f"basal_{i}")
            curr_basal.select_set(True)
            bpy.context.view_layer.objects.active = curr_basal
            deselect_object_vertices(curr_basal)
            bpy.ops.object.mode_set(mode='EDIT')
            # Connect basal region to valve.
            connect_valve_orifice_from_reference(context, "Mitral", valve_index = i, edges_reference = edges_reference) 
            basal_regions.append(curr_basal)
    return basal_regions

def connect_valve_orifice_reference(context, valve_mode, valve_index): # !!! ist diese funktion nötig -> gibt auch connect_valve_orifice
    """Connect orifices around valves with surrounding mesh nodes."""
    obj = bpy.context.active_object
    build_valve_surface(context, obj, valve_mode = valve_mode, ratio = 1, valve_index = valve_index) # Create valve interface nodes.
    bpy.ops.object.mode_set(mode='EDIT') # Change selection mode in edit mode for brige loop operator.
    selected_edges_before_indices, selected_edges_before_verts = get_selected_edges(obj)
    bpy.ops.mesh.looptools_bridge(cubic_strength=1, interpolation='linear', loft=False, loft_loop=False, min_width=100, mode='shortest', remove_faces=False, reverse=False, segments=1, twist=0)
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    selected_edges_after_indices, selected_edges_after_verts = get_selected_edges(obj)

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    
    new_edges = []
    new_edges_vert_indices = []
    for counter, e in enumerate(selected_edges_after_indices):
        if e not in selected_edges_before_indices:
            new_edges.append(e)
            new_edges_vert_indices.append(selected_edges_after_verts[counter])
        
    bpy.ops.object.mode_set(mode='OBJECT') 
    obj.select_set(False)
    return new_edges_vert_indices

def connect_valve_orifice_from_reference(context, valve_mode, valve_index, edges_reference): #!!! unfertig???
    """Connect the valve interface nodes with the valve orifice nodes from reference."""
    obj = bpy.context.active_object
    # Create exact valve nodes.
    build_valve_surface(context, obj, valve_mode = valve_mode, ratio = 1, valve_index = valve_index)
    bridge_edges_ventricle(obj, edges_reference)
    
def smooth_basal_region(context, voxel_size):
    """Smooth basal region."""
    bpy.ops.object.mode_set(mode='EDIT') 
    # Select all vertices in basal region except valve regions including the orifice edge loop and lower orifice loop.
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.object.vertex_group_set_active(group=str("lower_basal_edge_loop"))
    bpy.ops.object.vertex_group_deselect()
    bpy.ops.object.vertex_group_set_active(group=str("AV"))
    bpy.ops.object.vertex_group_deselect()
    bpy.ops.object.vertex_group_set_active(group=str("MV"))
    bpy.ops.object.vertex_group_deselect()
    bpy.ops.object.vertex_group_set_active(group=str("Aortic_orifice"))
    bpy.ops.object.vertex_group_deselect()
    bpy.ops.object.vertex_group_set_active(group=str("Mitral_orifice"))
    bpy.ops.object.vertex_group_deselect()
    # Merge close nodes and smooth them.
    if context.scene.bool_porous: bpy.ops.mesh.remove_doubles(threshold=voxel_size / 2, use_sharp_edge_from_normals=False, use_unselected=False)
    bpy.ops.mesh.vertices_smooth(factor=0.75, repeat=10)
    # Deselect valve orifice edge loops for a better smoothing transition between valves and basal region.
    bpy.ops.object.vertex_group_set_active(group=str("Aortic_orifice"))
    bpy.ops.object.vertex_group_select()
    bpy.ops.object.vertex_group_set_active(group=str("Mitral_orifice"))
    bpy.ops.object.vertex_group_select()
    # Merge and smooth again.
    if context.scene.bool_porous: bpy.ops.mesh.remove_doubles(threshold=voxel_size * 0.85, use_sharp_edge_from_normals=False, use_unselected=False)
    bpy.ops.mesh.vertices_smooth(factor=0.75, repeat=20)     
    bpy.ops.object.mode_set(mode='OBJECT') 

class MESH_OT_connect_apical_and_basal(bpy.types.Operator):
    """Connect apical and basal region of ventricle."""
    bl_idname = 'heart.connect_apical_and_basal'
    bl_label = 'Connect apical and basal region of ventricle.'
    def execute(self, context):
        if not mesh_connect_apical_and_basal(context): return {'CANCELLED'}
        return {'FINISHED'} 
        
def mesh_connect_apical_and_basal(context):
    """Connect apical and basal region of ventricle."""
    cons_print("Connecting apical and basal regions...")
    selected_objects = context.selected_objects
    # Initialize names for basal regions.
    if not context.scene.bool_porous: names = ["basal_0", "basal_1", "basal_2", "basal_3", "basal_4"]
    else: names = ["basal_0"]
    basal_regions = []
    # Set up basal regions so that the lower edge loop is selected.
    for name in names:
        if not name in bpy.data.objects: # Check if all necessary basal regions are present.
            cons_print(f"Missing following basal region: {name}")
            return False       
        else:  
            curr_basal = bpy.data.objects[name]
            basal_regions.append(curr_basal) # Add object to list of basal regions.
            # Unhide current basal region and use it as active object.
            curr_basal.hide_set(False)
            curr_basal.select_set(True)
            bpy.context.view_layer.objects.active = curr_basal
            # Select only lower basal edge loop vertex group.
            deselect_object_vertices(curr_basal)
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.object.vertex_group_set_active(group=str("lower_basal_edge_loop"))
            bpy.ops.object.vertex_group_select()
            bpy.ops.object.mode_set(mode='OBJECT')
            # Hide current basal region to improve solution speed as Blender does not need to render all objects at the same time.
            curr_basal.select_set(False)
            curr_basal.hide_set(True)
    # Create initial connection using a reference copy.
    reference = copy_object(bpy.types.Scene.reference_object_name, "reference")
    combine_apical_and_basal_region(context, basal_regions, reference, selected_objects)
    return True

def combine_apical_and_basal_region(context, basal_regions, reference, selected_objects):
    """Combine the two regions by copying and joining the basal region for each ventricle and connecting the orifice edge loops between these newly joined objects."""
    # Deselect (and hide) all objects.
    for obj in selected_objects: 
        obj.select_set(False)  
        obj.hide_set(True)
    reference.select_set(False)
    # Apply connecting operation for reference and save connecting edges used in the connection.
    prepare_geometry_for_bridging(context, reference, basal_regions[0]) # Prepare geometry for bridging by removing the original basal region and replacing it with the reconstructed basal region.
    edge_indices_bridge = bridge_edges_reference(context, reference) # Create initial connection between the upper apical and lower basal edge loop.
    inset_faces_smooth(context) # Refine connection by separating long connection faces into more uniformly sized faces.
    edge_indices_triangulate = triangulate_connection(True, reference, ref_edge_indices=[]) # Triangulate connection faces saving newly created edges.
    bpy.data.objects.remove(reference, do_unlink=True) # Cleanup: Remove reference object.
    # Compute the frame of the end diastole. Necessary for interpolated mitral valve.
    frame_EDV = round(context.scene.time_diastole / context.scene.time_rr *  context.scene.frames_ventricle) 
    # Apply connecting-operation for remaining ventricle geometries.
    for counter, obj in enumerate(selected_objects):
        basal = basal_regions[get_valve_state_index(context, counter, frame_EDV)] # Choose basal region.
        # Apply connecting operation from reference.
        prepare_geometry_for_bridging(context, obj, basal) 
        bridge_edges_ventricle(obj, edge_indices_bridge)
        inset_faces_smooth(context)
        # Remove faces before triangulation.
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_more()
        bpy.ops.mesh.delete(type='ONLY_FACE') 
        bpy.ops.object.mode_set(mode='OBJECT')
        # Triangulate mesh.
        triangulate_connection(False, obj, edge_indices_triangulate)        
        # Add faces onto triangulation.
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.edge_face_add()
        bpy.ops.object.mode_set(mode='OBJECT')
        # Smooth connection.
        smooth_connection_and_basal_region(context, obj)
        obj.hide_set(True)  
    for obj in selected_objects: obj.hide_set(False) # Cleanup: Unhide objects.

def get_valve_state_index(context, counter, frame_EDV):
    """Return the index of the basal region to be used for the given timestep."""
    if context.scene.bool_porous: return 0
    # Define the opening state of the mitral valve during multiple timeframes.
    begin_mvo = 4
    frames_mv_1 = [begin_mvo, frame_EDV-1] #  Fill with indices.
    frames_mv_2 = [begin_mvo + 1, frame_EDV-2]
    frames_mv_3 = [begin_mvo + 2, frame_EDV-3]
    frames_mv_4 = range(begin_mvo + 3, (frame_EDV-3))
    # Return index.
    if counter in frames_mv_1: return 1
    if counter in frames_mv_2: return 2
    if counter in frames_mv_3: return 3
    if counter in frames_mv_4: return 4
    else: return 0

def prepare_geometry_for_bridging(context, obj, final_basal_region):
    """Prepare the individual ventricle geometries for the bridging process. The original basal region is removed and the remaining apical region is joined into one object with the reconstructed basal region."""
    current_basal = copy_object(final_basal_region.name, 'temp') # Copy basal region for current object.
    # Change selection such that the copy of the basal region is merged into the input-object (obj).
    current_basal.select_set(False)
    obj.hide_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    deselect_object_vertices(obj)
    # Cut basal part of the ventricle.
    dissolve_edge_loops(context, obj)
    # Combine both geometries.
    current_basal.select_set(True)
    bpy.ops.object.join()

def bridge_edges_reference(context, reference):
    """Connect basal with apical part of the reference ventricle."""
    bpy.ops.object.mode_set(mode='EDIT') # Need to switch to edit mode. 
    selected_edges_before_indices, selected_edges_before_verts = get_selected_edges(reference) # Collect edge indices before operation.
    # Connect apical and basal region.
    bpy.context.tool_settings.mesh_select_mode = (False, True, False) # Activate edge mode in edit mode.
    bpy.ops.mesh.looptools_bridge(cubic_strength=1, interpolation='linear', loft=False, loft_loop=False, min_width=100, mode='shortest', remove_faces=False, reverse=False, segments=1, twist=context.scene.connection_twist) # Command used to connect two edge loops.
    bpy.context.tool_settings.mesh_select_mode = (True, False, False) # Return to vertex mode in edit mode.
    selected_edges_after_indices, selected_edges_after_verts = get_selected_edges(reference) # Save edge indices after operation.
    # Compare edges before and after operation and only keep the deviating edges and save their vertex indices in a list.
    new_edges, new_edges_vert_indices = ([] for i in range(2))
    for counter, e in enumerate(selected_edges_after_indices):
        if e not in selected_edges_before_indices:
            new_edges.append(e)
            new_edges_vert_indices.append(selected_edges_after_verts[counter])
    bpy.ops.object.mode_set(mode='OBJECT')
    return new_edges_vert_indices

def bridge_edges_ventricle(obj, new_edges_vert_indices): 
    """Connect basal with apical part of ventricle."""
    deselect_object_vertices(obj) # Deselect object vertices. This is necessary to add faces later in the function call.
    # Transfer data to edit-mode.
    bpy.ops.object.mode_set(mode='EDIT') 
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    for a, b in new_edges_vert_indices: # Create connecting edges between all vertex pairs a and b and keep them selected.
        bm.edges.new((bm.verts[a], bm.verts[b]))
        bm.verts[a].select = True
        bm.verts[b].select = True
    # Create faces between all selected connecting edges (Wireframe to Surface-mesh).
    bpy.ops.object.mode_set(mode='OBJECT') # Necessary switch between object and edit mode to update blender-object.
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.edge_face_add() # Create faces between connection edges.
    bpy.ops.object.mode_set(mode='OBJECT') 

def inset_faces_smooth(context):
    """Create new vertices along the connection between apical and basal region. This subdivision aims to more equally space the height and width of these faces."""
    distance = context.scene.min_valves - context.scene.max_apical # Distance between lowest valve node and highest apical point of all ventricles after basal removal in z-direction.
    bpy.ops.object.mode_set(mode='EDIT')
    for counter, value in enumerate(range(context.scene.inset_faces_refinement_steps)):
        inset_thickness = distance / (2 * 2**(counter)) # Reduce the offset between newly added edge_loops by the factor 4.
        bpy.ops.mesh.inset(thickness= inset_thickness, depth=0, use_select_inset=True) # Insert faces along the bridge between apical and basal.
        bpy.ops.mesh.vertices_smooth(factor=0.5, repeat=1 + counter)  # Smooth connection between apical and basal region including the newly added faces.
    bpy.ops.object.mode_set(mode='OBJECT')
  
def triangulate_connection(bool_ref, obj, ref_edge_indices):
    """Triangulate connection between apical and basal region."""
    edges_vert_indices_tri = []
    if bool_ref: # Initial triangulation as a reference.
        bpy.ops.object.mode_set(mode='EDIT') 
        selected_edges_before_indices, selected_edges_before_verts = get_selected_edges(obj) # Collect edge indices before operation.
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY') # Triangulate connection.
        selected_edges_after_indices, selected_edges_after_verts = get_selected_edges(obj) # Collect edge indices after operation.
        # Compare edges before and after operation and only keep the new edges and save their corresponding vertex indices in a list.
        for counter, e in enumerate(selected_edges_after_indices):
            if e not in selected_edges_before_indices:
                edges_vert_indices_tri.append(selected_edges_after_verts[counter])
        bpy.ops.object.mode_set(mode='OBJECT') 
    else: # Apply triangulation from reference.
        deselect_object_vertices(obj) # Deselect all faces to add faces later.
        bpy.ops.object.mode_set(mode='EDIT') 
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        # Create connecting edge between a and b
        for a, b in ref_edge_indices:
            bm.edges.new((bm.verts[a], bm.verts[b]))
            bm.verts[a].select = True
            bm.verts[b].select = True
        # Create faces between connecting edges
        bpy.ops.object.mode_set(mode='OBJECT') # Necessary to update geometry in Blender.
        bpy.ops.object.mode_set(mode='EDIT') 
        bpy.ops.object.mode_set(mode='OBJECT') 
    return edges_vert_indices_tri

def smooth_connection_and_basal_region(context, obj): 
    """Smooth basal ventricle region excluding the valves."""
    deselect_object_vertices(obj) # Reset node selection.
    # Select all vertices between the lowest valve vertex and the highest basal region vertex.
    bm = transfer_data_to_mesh(obj)
    for v in bm.verts: # Select all vertices above highest apical vertex in z-direction.
        vertex_coords = obj.matrix_world @ v.co 
        if vertex_coords[2] > context.scene.max_apical: v.select = True
    bm.to_mesh(obj.data)
    bpy.ops.object.mode_set(mode='EDIT')  
    # Select all nodes inside the connection.
    bpy.ops.object.vertex_group_set_active(group=str("lower_basal_edge_loop"))
    bpy.ops.object.vertex_group_select()  
    # Select edge loops below the connection. Selecting all apical nodes would greatly shrink the ventricle volume in that region.
    for i in range(5):  # Iteratively smooth the selected nodes. This especially smooths the transition between connection and apical nodes.
        bpy.ops.mesh.select_more() # Select edge loops until reaching an edgeloop, that was not subdivided during the removal of the basal region.
        # Exclude valve nodes in the selection process.
        bpy.ops.object.vertex_group_set_active(group=str("AV"))
        bpy.ops.object.vertex_group_deselect()
        bpy.ops.object.vertex_group_set_active(group=str("MV"))
        bpy.ops.object.vertex_group_deselect()
        # Continuously weaker smoothing. As hard smoothing creates kinks between smoothed nodes and unsmoothed nodes.
        bpy.ops.mesh.vertices_smooth(factor=0.5, repeat=6-i)
    bpy.ops.object.mode_set(mode='OBJECT')

class MESH_OT_Ventricle_Sort(bpy.types.Operator): 
    """Sort ventricles by volume starting with ESV."""
    bl_idname = 'heart.sort_ventricles'
    bl_label = 'Sort ventricles by volume starting with ESV.'
    def execute(self, context):
        if not sort_ventricles(context, context.selected_objects): return{'CANCELLED'}
        return{'FINISHED'}

def sort_ventricles(context, selected_objects):
    """Sort ventricles by volume starting with minimal volume (End-systolic volume)."""
    volumes = get_volumes(selected_objects, True)
    val, idx = min((val, idx) for (idx, val) in enumerate(volumes))
    # Free up names by renaming the objects to temporary names.
    for counter, obj in enumerate(selected_objects): obj.name = f"temp_ven_{counter}" 
    # Re-arrange names of ventricles.
    for old_index, obj in enumerate(selected_objects): 
        if old_index < idx: new_index = old_index - idx + len(volumes) # If index would go below 0 revert to the other side of the list.
        else:new_index = old_index - idx 
        obj.name = f"ventricle_{new_index}" # Rename selected objects.

class MESH_OT_Ventricle_Interpolation(bpy.types.Operator): 
    """Interpolate ventricle geometry to a larger amount of timesteps."""
    bl_idname = 'heart.ventricle_interpolation'
    bl_label = 'Interpolate selected objects.'
    def execute(self, context):
        if not interpolate_ventricle(context): return{'CANCELLED'}
        return{'FINISHED'}
    
def interpolate_ventricle(context):
    """Interpolate selected objects to n(frames_ventricle) objects."""
    cons_print("Interpolating ventricle geometries...")
    selected_objects = context.selected_objects
    for counter, obj in enumerate(selected_objects): obj.name = f"z_ven_{counter}" # Rename selected objects.
    # Compute timesteps before and after interpolation.
    timestep_original = context.scene.time_rr / len(selected_objects) 
    timestep_int = context.scene.time_rr / (context.scene.frames_ventricle) 
    int_objects = []
    # Copy first object and add it to the end of the object list to be able to interpolate between the last timestep and the first.
    initial_obj = copy_object(selected_objects[0].name, f"z_ven_{len(selected_objects)}")
    selected_objects.append(initial_obj)
    for i in range(context.scene.frames_ventricle): 
        if i == 0: # Exceptions for first object, since it doesn not have to be interpolated.
           int_objects.append(copy_object(selected_objects[0].name, 'ventricle_0')) 
           continue # Skip further functionality and fo straight to the second iteration.
        # Compute indices of the objects used in the current interpolation.
        current_time = (i) * timestep_int
        object_one_index = math.floor(current_time / timestep_original) # Find first object in the current timestep.
        object_one = selected_objects[object_one_index]
        object_two = selected_objects[object_one_index + 1]
        # Interpolate chosen objects using an interpolation factor defining how strongly each object influences the resulting object after interpolation.
        int_factor = (current_time - (object_one_index * timestep_original)) / timestep_original 
        int_obj = interpolate_two_object(object_one, object_two, int_factor, new_obj_index=i) 
        # Add the interpolated object to the list of interpolated objects.
        if not int_obj: # Exception when object could not be interpolated.
            cons_print(f"Problem during interpolation of timestep: {i}.")
            return False
        int_obj.select_set(True)
        int_objects.append(int_obj)
    for i in range(len(selected_objects)): bpy.data.objects.remove(bpy.data.objects[f"z_ven_{i}"], do_unlink=True) # Cleanup: Delete original objects.
    return int_objects

def interpolate_two_object(object_one, object_two, int_factor, new_obj_index):
    """Interpolate two objects using an interpolation factor."""
    # Get list of node coordinates for both objects.
    coords_one = get_object_vertices(object_one)
    coords_two = get_object_vertices(object_two)
    if len(coords_one) != len(coords_two): # Check if both objects have the same amount of nodes.
        cons_print("Tried interpolating objects with deviating amount of nodes.")
        return False
    # Compute interpolation for all nodes.
    int_coords = []
    for index in range(len(coords_one)):
        vec_int = coords_one[index] + (coords_two[index] - coords_one[index]) * int_factor # Interpolation of a single node.
        int_coords.append(vec_int)
    # Create new object by copying the first object and change the nodes coordinates to the interpolated node coordinates.
    int_obj = copy_object(object_one.name, f"ventricle_{new_obj_index}") # Copy object_one to create object between original timeframes.
    bpy.context.view_layer.objects.active = int_obj # Set new object as active object.
    change_object_vertices(int_obj, int_coords) # Change node coordinates.
    return int_obj
    
def get_object_vertices(obj):  
    """Return list of vertex coordinates of an object."""
    if obj.mode == 'EDIT': bpy.ops.object.mode_set() # Change to object mode if in edit mode to be able to use obj.data.vertices.
    coords = []
    for v in obj.data.vertices: coords.append(obj.matrix_world @ v.co)
    return coords

def change_object_vertices(obj, vertices):
    """Change object vertices with a list of vertices."""
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data) # Transfer data to mesh.
    for v in bm.verts: v.co = vertices[v.index] # Change vertices.
    bpy.ops.object.mode_set(mode='OBJECT')

class MESH_OT_Add_Vessels_Valves(bpy.types.Operator):
    """Create objects for aorta, atrium and valves."""
    bl_idname = 'heart.add_vessels_valves'
    bl_label = 'Create objects for aorta, atrium and valves.'
    def execute(self, context):
        add_vessels_and_valves(context)
        return{'FINISHED'}
    
def add_vessels_and_valves(context): 
    """Create objects for aorta, atrium and valves."""
    add_aorta(context)
    add_atrium(context)
    create_porous_valve_zones(context, 'Aortic', ['por_AV_imperm', 'por_AV_perm', 'por_AV_res']) 
    if context.scene.bool_porous: create_porous_valve_zones(context, 'Mitral', ['por_MV_imperm', 'por_MV_perm', 'por_MV_res']) # Porous mitral valve are only inserted in the porous medium approach. For the interpolated mitral valve approach the mitral valve becomes part of the ventricle.

class MESH_OT_Quick_Recon(bpy.types.Operator):
    """Quick geometrical reconstruction of all ventricles containing all steps of the reconstruction algorithm in one execution."""
    bl_idname = 'heart.quick_recon'
    bl_label = 'Quick geometrical reconstruction of all ventricles containing all steps of the reconstruction algorithm in one execution.'
    def execute(self, context):
        if not interpolate_ventricle(context): return{'CANCELLED'} # Interpolate ventricle geometry.
        if not mesh_create_basal(context): return{'CANCELLED'}# Operations to create basal region of the ventricle containing valve orifices.
        if not mesh_connect_apical_and_basal(context): return {'CANCELLED'} # Connect apical regions with corresponding bassal regions.
        add_vessels_and_valves(context) # Add surrounding objects including aorta, atrium and valves.
        cleanup_basal_region(context) # Cleanup: Delete basal regions.
        return{'FINISHED'} 

def cleanup_basal_region(context):
    """Remove basal regions as part of the reconstruction pipeline cleanup."""
    if context.scene.bool_porous: amount_basal_regions=1
    else: amount_basal_regions=5
    for i in range(amount_basal_regions): bpy.data.objects.remove(bpy.data.objects["basal_" + str(i)], do_unlink=True)
    
class MESH_DEV_volumes(bpy.types.Operator):  
    """Compute volumes of selected objects."""
    bl_idname = 'heart.dev_volumes'
    bl_label = 'Compute volumes of selected objects.'
    def execute(self, context):
        get_volumes(bpy.context.selected_objects, True)
        return{'FINISHED'}

def get_volumes(objects, bool_print):
    """Print each volume and surface area of a given list of objects."""
    volumelist = []
    for obj in objects:
        volume, area = compute_volume_area(obj)
        volumelist.append(volume)
        if bool_print: cons_print(f"{obj.name} with volume: {round(volume/1000, 4)} ml and surface area: {round(area/100, 4)} mm^2.")
    return volumelist

def compute_volume_area(obj):
    """Return volume and surface area of object of object."""  
    # Transfer object into mesh.
    bm = transfer_data_to_mesh(obj)
    # Compute volume and surface area.
    volume = bm.calc_volume(signed=True) 
    area = sum(f.calc_area() for f in bm.faces)
    return volume, area

class MESH_DEV_indices(bpy.types.Operator):
    """Print indices of selected nodes and print the total number of selected nodes."""
    bl_idname = 'heart.dev_indices'
    bl_label = 'Print indices of selected nodes and print the total number of selected nodes.'
    def execute(self, context):
        if not get_selected_objects_vertices(context): return{'CANCELLED'}
        return{'FINISHED'}
    
def get_selected_objects_vertices(context):
    """Return indices of currently selected vertices."""
    if bpy.context.mode != 'EDIT': bpy.ops.object.mode_set(mode='EDIT') # Toggle to edit mode.
    obj = context.object
    bm = bmesh.from_edit_mesh(obj.data)
    verts = []
    for v in bm.verts:
        if v.select:
            vertice_coords = obj.matrix_world @ v.co # Transform to global coordinate system.
            cons_print(f"Vertex {v.index} at ({round(vertice_coords.x, 2)}|{round(vertice_coords.y, 2)}|{round(vertice_coords.z, 2)})")
            verts.append(v)
    cons_print(f"Amount of selected vertices: {len(verts)}")
    return verts

class MESH_DEV_edge_index(bpy.types.Operator): 
    """Print edge index between two selected indices."""
    bl_idname = 'heart.dev_check_edges'
    bl_label = 'Print edge index between two selected indices.'
    def execute(self, context):
        if not check_edge(context): return{'CANCELLED'}
        return{'FINISHED'}

def check_edge(context):
    """Print edge index between two selected indices."""
    # Read selected vertices.
    obj = context.edit_object
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    if hasattr(bm.verts, "ensure_lookup_table"): bm.verts.ensure_lookup_table()
    verts = [v for v in bm.verts if v.select] # Get selected vertices.
    # Only two vertices may be selected.
    if len(verts) != 2:
        cons_print(f"Select exactly two vertices.")
        return False
    # Check edge connectivity between vertices and print out result.
    V1 = verts[0] # Define vertices one and two.
    V2 = verts[1]
    for e in V1.link_edges: 
        if e.other_vert(V1) is V2:
            cons_print(f"Vertex indices : {V1.index}, {V2.index} with edge index: {e.index}.")
            return e.index
    cons_print(f"Nodes not connected.")
    return False

class MESH_DEV_check_node_connectivity(bpy.types.Operator):
    """Check node connectivity between all selected objects."""
    bl_idname = 'heart.dev_check_node_connectivity'
    bl_label = 'Check node connectivity between all selected objects.'
    def execute(self, context):
        check_node_connectivity(context)
        return{'FINISHED'}

def check_node_connectivity(context):
    """Check node-connectivity between all selected objects."""
    vertices, edges, faces = ([] for i in range(3))
    for i, obj in enumerate(context.selected_objects):
    # Initialize the list of vertices, edges and faces each with their respective connecting vertices.
        if i == 0:
            cons_print(f"Initial object: {obj.name}")
            for v in obj.data.vertices: vertices.append(v.index)
            for e in obj.data.edges: edges.append([e.index, e.vertices[0] , e.vertices[1]])
            for f in obj.data.polygons: faces.append([f.index, f.vertices[0], f.vertices[1], f.vertices[2]])
        else:
    # Checks for the length of vertices, edges and faces.
            # Check if the amount of vertices matches.
            if len(obj.data.vertices) != len(vertices):
                cons_print(f"Object: {obj.name} has mismatching amount of vertices.")
                return False
            # Check if the amount of edges matches.
            if len(obj.data.edges) != len(edges):
                cons_print(f"Object: {obj.name} has mismatching amount of edges.")
                return False
            # Check if the amount of faces matches.
            if len(obj.data.polygons) != len(faces):
                cons_print(f"Object: {obj.name} has mismatching amount of faces.")
                return False
    # Connectivity checks. 
            # Face-connectivity check.
            for counter, f in enumerate(obj.data.polygons):
                if not (faces[counter][0] == f.index and faces[counter][1]== f.vertices[0] and faces[counter][2] ==  f.vertices[1] and faces[counter][3] ==  f.vertices[2]):
                    cons_print(f"Face mismatch in object: {obj.name} at face {f.index} with face-center at {f.center}.") 
                    deselect_object_vertices(obj)
                    f.select = True # Select problematic face.
                    return False
            # Edge-connectivity check.
            for counter, e in enumerate(obj.data.edges):
                if not (edges[counter][0] == e.index and edges[counter][1]== e.vertices[0] and edges[counter][2] ==  e.vertices[1]):
                    cons_print(f"Edge mismatch in object: {obj.name} at edge {e.index}.")
                    deselect_object_vertices(obj)
                    e.select = True # Select problematic edge.
                    return False  
    cons_print("Node-connectivity matched for all selected elements.")
    return True

class PANEL_Position_Ventricle(bpy.types.Panel):
    bl_label = "Ventricle position (mm)"
    bl_idname = "PT_Ventricle"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Herz'
    bl_option = {'DEFALUT_CLOSED'}
    def draw(self, context):
        layout = self.layout 
        row = layout.row()
        row.prop(context.scene, 'pos_top', text="Basal (Top)")  
        row = layout.row()
        props = layout.operator('heart.get_point', text= "Select basal node")
        props.point_mode = "Top"
        row = layout.row()
        row.prop(context.scene, 'pos_bot', text="Apex (bottom)")    
        row = layout.row()
        props = layout.operator('heart.get_point', text= "Select apex node")
        props.point_mode = "Bot"
        row = layout.row()
        row.prop(context.scene, 'pos_septum', text="Septum (side)")  
        row = layout.row()
        props = layout.operator('heart.get_point', text= "Select node at septum")
        props.point_mode = "Septum"  
        row = layout.row()
        row.label(text= "Rotation and cutting") 
        row = layout.row()
        layout.operator('heart.ventricle_rotate', text= "Translate and rotate", icon = 'CON_ROTLIKE')

class PANEL_Interpolation(bpy.types.Panel):
    bl_label = "Interpolation"
    bl_idname = "PT_Interpolation"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Herz'
    bl_option = {'DEFALUT_CLOSED'}
    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.prop(context.scene, 'time_rr', text="Time RR-duration") 
        row = layout.row()
        row.prop(context.scene, 'time_diastole', text="Time diastole") 
        row = layout.row()
        layout.operator('heart.ventricle_interpolation', text= "Interpolate ventricle", icon = 'IPO_EASE_IN')

class PANEL_Valves(bpy.types.Panel):
    bl_label = "Valve options"
    bl_idname = "PT_Valves"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Herz'
    bl_option = {'DEFALUT_CLOSED'}
    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.label(text= "Mitral valve", icon = 'META_ELLIPSOID')  
        row = layout.row()
        row.prop(context.scene, 'translation_mitral', text="Translation(mm)", icon='META_ELLIPSOID')
        row = layout.row()
        row.prop(context.scene, 'angle_mitral', text="Angle", icon='META_ELLIPSOID')

        row = layout.row()
        row.prop(context.scene, 'mitral_radius_long', text="Long mitral radius", icon='META_ELLIPSOID')
        row = layout.row()
        row.prop(context.scene, 'mitral_radius_small', text="Small mitral radius")

        row = layout.row()
        row.label(text= "Aortic valve", icon = 'MESH_CIRCLE')   
        row = layout.row()
        row.prop(context.scene, 'translation_aortic', text="Translation(mm)")
        row = layout.row()
        row.prop(context.scene, 'angle_aortic', text="Angle")
        row = layout.row()
        row.prop(context.scene, 'aortic_radius', text="Aortic radius", icon='META_BALL')
        row = layout.row()
        layout.prop(context.scene, "bool_porous", text="Porous mitral valve")
        row = layout.row()
        layout.operator('heart.build_valve',  text= "Add valve interface nodes", icon = 'PROP_OFF')
        row = layout.row()
        layout.operator('heart.support_struct',  text= "Build support structure around valves", icon = 'PROP_ON')
                
class PANEL_Poisson(bpy.types.Panel):
    bl_label = "Poisson"
    bl_idname = "PT_Poisson"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Herz'
    bl_option = {'DEFALUT_CLOSED'}
    def draw(self, context):
        layout = self.layout      
        row = layout.row()
        row.operator('heart.cut_edge_loops', text= "Remove edge loops from top position", icon = 'LIBRARY_DATA_OVERRIDE') 
        row = layout.row()
        row.prop(context.scene, 'remove_basal_threshold', text="Threshold for basal region removal") 
        row = layout.row()
        row.operator('heart.remove_basal', text= "Remove basal region", icon = 'LIBRARY_DATA_OVERRIDE') 
        row = layout.row()
        layout.operator('heart.poisson', text= "Apply Poisson surface reconstruction", icon = 'PROP_ON')
        row = layout.row()
        layout.operator('heart.create_valve_orifice', text= "Create valve orifices", icon = 'ALIASED')
        row = layout.row()
        layout.operator('heart.connect_valve', text= "Connect valves to orifices", icon = 'ANTIALIASED')
   
class PANEL_Setup_Variables(bpy.types.Panel):
    bl_label = "Algorithm setup variables"
    bl_idname = "PT_test"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Herz'
    bl_option = {'DEFALUT_CLOSED'}
    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.prop(context.scene, 'amount_of_cuts', text="Amount of cut edge loops")  
        row = layout.row()
        row.prop(context.scene, 'poisson_depth', text="Depth of poisson reconstruction algorithm") 
        row = layout.row()
        layout.prop(context.scene, "inset_faces_refinement_steps", text="Refinement steps for insetting faces.")
        row = layout.row()
        layout.prop(context.scene, "connection_twist", text="Twist during connecting algorithm")

class PANEL_Objects(bpy.types.Panel):
    bl_label = "Surrounding objects"
    bl_idname = "PT_surroundings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Herz'
    bl_option = {'DEFALUT_CLOSED'}
    def draw(self, context):
        layout = self.layout
        row = layout.row()
        layout.operator('heart.add_porous_zones', text= "Add porous zone valves", icon = 'ALIGN_FLUSH')
        row = layout.row()
        layout.operator('heart.add_atrium', text= "Add atrium", icon = 'CURSOR')
        row = layout.row()
        layout.operator('heart.add_aorta', text= "Add aorta", icon = 'MESH_CYLINDER')

class PANEL_Reconstruction(bpy.types.Panel):
    bl_label = "Ventricle reconstruction"
    bl_idname = "PT_Reconstruction"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Herz'
    bl_option = {'DEFALUT_CLOSED'}
    def draw(self, context):
        layout = self.layout
        row = layout.row()
        layout.operator('heart.ventricle_interpolation', text= "Interpolate ventricle", icon = 'IPO_EASE_IN')
        row = layout.row()
        layout.prop(context.scene, "bool_porous", text="Porous mitral valve")
        row = layout.row()
        layout.operator('heart.create_basal', text= "Create basal region", icon = 'SPHERECURVE')
        row = layout.row()
        layout.operator('heart.connect_apical_and_basal', text= "Connect basal and apical parts", icon = 'ORPHAN_DATA')
        row = layout.row()
        layout.operator('heart.add_vessels_valves', text= "Add atrium, aorta and valves", icon = 'HANDLE_AUTO')
        row = layout.row()
        layout.operator('heart.quick_recon', text= "Quick reconstruction", icon = 'HEART')

class PANEL_Dev_tools(bpy.types.Panel):
    bl_label = "Development tools"
    bl_idname = "PT_Dev_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Herz'
    bl_option = {'DEFALUT_CLOSED'}
    def draw(self, context):
        layout = self.layout
        row = layout.row()
        layout.operator('heart.dev_volumes', text= "Compute volumes", icon = 'HOME')
        row = layout.row()
        layout.operator('heart.sort_ventricles', text= "Sort volumes", icon = 'HOME')
        row = layout.row()
        layout.operator('heart.dev_indices', text= "Get vertex indices", icon = 'THREE_DOTS')
        row = layout.row()
        layout.operator('heart.dev_check_edges', text= "Get edge index", icon = 'ARROW_LEFTRIGHT')
        row = layout.row()
        layout.operator('heart.dev_check_node_connectivity', text= "Node-connectivity check", icon = 'CHECKMARK')
        row = layout.row()

classes = [
    PANEL_Position_Ventricle,
    PANEL_Interpolation, PANEL_Valves, PANEL_Poisson, PANEL_Objects, PANEL_Reconstruction, PANEL_Setup_Variables,  PANEL_Dev_tools, MESH_OT_get_node, MESH_OT_ventricle_rotate, MESH_OT_poisson, MESH_OT_build_valve, MESH_OT_create_valve_orifice, 
    MESH_OT_support_struct, MESH_OT_connect_valves, MESH_OT_cut_edge_loops, MESH_OT_Add_Atrium, MESH_OT_Add_Aorta, MESH_OT_Porous_zones, MESH_OT_Ventricle_Sort, MESH_OT_Quick_Recon, MESH_OT_new_remove_basal,
    MESH_OT_create_basal, MESH_OT_connect_apical_and_basal, MESH_OT_Ventricle_Interpolation, MESH_OT_Add_Vessels_Valves, MESH_DEV_volumes, MESH_DEV_indices, MESH_DEV_edge_index, MESH_DEV_check_node_connectivity,
]
  
def register():
    # Position variables.
    bpy.types.Scene.pos_top = bpy.props.FloatVectorProperty(name="Top position", default = (0,0,1))
    bpy.types.Scene.pos_bot = bpy.props.FloatVectorProperty(name="Top position", default = (0,0,0))
    bpy.types.Scene.pos_septum = bpy.props.FloatVectorProperty(name="Top position", default = (0,1,0))
    bpy.types.Scene.top_index = bpy.props.IntProperty(name="Index of top position", default = 0)
    # Cutting plane variables.
    bpy.types.Scene.height_plane = bpy.props.FloatProperty(name="Height(z-value) of intersection plane", default=40,  min = 0.01)
    bpy.types.Scene.min_valves = bpy.props.FloatProperty(name="Minimal z-value of valves", default=45)
    bpy.types.Scene.max_apical = bpy.props.FloatProperty(name="Maximal z-value of apical region after cutting", default=40)
    bpy.types.Scene.amount_of_cuts = bpy.props.IntProperty(name="Amount of edge loop cuts from top position", default=10,  min = 2)

    bpy.types.Scene.remove_basal_threshold = bpy.props.IntProperty(name="Threshold for the removal of the basal region.", default=25,  min = 0)
    # Possion algorithm.
    bpy.types.Scene.poisson_depth = bpy.props.IntProperty(name="Depth of possion algorithm", default=10,  min = 1)
    # Aortic valve.
    bpy.types.Scene.aortic_radius = bpy.props.FloatProperty(name="aortic_radius", default=2,  min = 0.01)
    bpy.types.Scene.translation_aortic = bpy.props.FloatVectorProperty(name="Aortic valve translation", default = (0,0,1))
    bpy.types.Scene.angle_aortic = bpy.props.FloatVectorProperty(name="Aortic valve rotation", default = (0,0,0))
    # Mitral valve.
    bpy.types.Scene.mitral_radius_long = bpy.props.FloatProperty(name="mitral_radius_long", default=6,  min = 0.01)
    bpy.types.Scene.mitral_radius_small = bpy.props.FloatProperty(name="mitral_radius_small", default=3,  min = 0.01)
    bpy.types.Scene.translation_mitral = bpy.props.FloatVectorProperty(name="Aortic valve translation", default = (0,0,1))
    bpy.types.Scene.angle_mitral = bpy.props.FloatVectorProperty(name="Aortic valve rotation", default = (0,0,0))
    bpy.types.Scene.bool_porous = bpy.props.BoolProperty(name="Porous approach for mitral valve?", default = False)
    # Interpolation variables.
    bpy.types.Scene.time_rr = bpy.props.FloatProperty(name="Time RR-duration", default=0.6,  min = 0.01)
    bpy.types.Scene.time_diastole = bpy.props.FloatProperty(name="Time diastole", default=0.35,  min = 0.01) # !!!Compute automatically using the volumes and rr-duration. Automatische sortierung mit ESV am anfang waere auch gut
    bpy.types.Scene.frames_ventricle = bpy.props.IntProperty(name="Amount of frames ventricle after interpolation", default=10,  min = 10)
    # Connection variables.
    bpy.types.Scene.reference_object_name = bpy.props.StringProperty(name="Name of the reference object", default = "ventricle_0")
    bpy.types.Scene.inset_faces_refinement_steps = bpy.props.IntProperty(name="Refinement steps when insetting faces in the connection algorithm.", default=1, min=1)
    bpy.types.Scene.connection_twist = bpy.props.IntProperty(name="Twist for bridging algorithm in connection.", default=0)
    # Register UI-classes for Panels and functions.
    for c in classes: bpy.utils.register_class(c)
    
def unregister(): # Unregister classes.
    for c in classes: bpy.utils.unregister_class(c)
    
if __name__ == '__main__': register()  