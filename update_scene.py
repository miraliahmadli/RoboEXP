def update_scene(robo_memory, robo_percept, 
                 observations, object_level_labels, 
                 first_scene=False, visualize=False):
    observation_attributes = robo_percept.get_attributes_from_observations(
        observations, visualize=visualize
    )
    if first_scene:
        # for initial update of scene graph
        scene_graph_option = None
    else:
        scene_graph_option = {}
        scene_graph_option['type'] = 'check'
        scene_graph_option['old_instances'] = robo_memory.memory_instances

    filter_masks = {"wrist": None}
    direct_move = None

    robo_memory.update_memory(
        observations,
        observation_attributes,
        object_level_labels,
        direct_move=direct_move,
        filter_masks=filter_masks,
        extra_alignment=False,
        update_scene_graph=True,
        scene_graph_option=scene_graph_option,
        visualize=visualize,
    )

    return robo_memory


if __name__ == "__main__":
    from roboexp import RoboMemory, RoboPercept
    # Set the labels
    object_level_labels = [
        "microwave",
        "toaster",
        "mug",
        "cabinet",
        "plate",
        "countertop",
        "LED",
        "buttons",
    ]
    part_level_labels = ["handle"]

    grounding_dict = (
        " . ".join(object_level_labels) + " . " + " . ".join(part_level_labels)
    )
    # Initialize the perception module
    robo_percept = RoboPercept(grounding_dict=grounding_dict, lazy_loading=False)
    robo_memory = RoboMemory(
        lower_bound=[-5, -5, -5],
        higher_bound=[3, 3, 3],
        voxel_size=0.02,
        real_camera=True,
        base_dir=None,
        similarity_thres=0.95,
        iou_thres=0.01,
    )
    
    observations = []
    # Update the scene
    robo_memory = update_scene(robo_memory, robo_percept, 
                               observations, object_level_labels, 
                               first_scene=True, visualize=False)

