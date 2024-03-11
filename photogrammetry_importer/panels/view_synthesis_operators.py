import sys
import subprocess
import os
import numpy as np
import bpy
from mathutils import Matrix
from tempfile import NamedTemporaryFile
from bpy_extras.io_utils import ExportHelper


from photogrammetry_importer.utility.np_utility import (
    invert_transformation_matrix,
)
from photogrammetry_importer.blender_utility.retrieval_utility import (
    get_selected_camera,
    get_scene_animation_indices
)
from photogrammetry_importer.blender_utility.logging_utility import log_report
from photogrammetry_importer.importers.camera_utility import (
    load_background_image,
    get_computer_vision_camera,
)
from photogrammetry_importer.file_handlers.instant_ngp_file_handler import (
    InstantNGPFileHandler,
)
from photogrammetry_importer.process_communication.subprocess_command import (
    create_subprocess_command,
)
from photogrammetry_importer.process_communication.file_communication import (
    read_np_array_from_file,
)


def run_view_synth(scene, save_to_dp=None, op=None):
    log_report(
        "INFO", "Compute view synthesis for current camera: ...", op
    )

    command, temp_json_file, temp_array_file = create_instant_ngp_cmd(scene, output_dp=save_to_dp, op=op)

    camera_relative_to_anchor, centroid_shift = shift_selected_camera_relative_to_anchor(scene)

    # Call before executing the child process
    InstantNGPFileHandler.write_instant_ngp_file(
        temp_json_file.name,
        [camera_relative_to_anchor],
        ref_centroid_shift=centroid_shift,
    )

    child_process = subprocess.Popen(command)
    child_process.communicate()

    show_image_in_blender(temp_array_file, get_selected_camera())

    cleanup_tmp_files(temp_json_file, temp_array_file)

    log_report(
        "INFO", "Compute view synthesis for current camera: Done", op
    )
    return {"FINISHED"}

class RunViewSynthesisOperator(bpy.types.Operator):
    """An Operator to use the camera to render the NeRF Model, saving the output as Blender Image"""

    bl_idname = "photogrammetry_importer.run_view_synthesis"
    bl_label = "Save as Blender Image"
    bl_description = "Export camera properties to json and run the given script. The result is displayed as Blender Image"

    @classmethod
    def poll(cls, context):
        """Return the availability status of the operator."""
        cam = get_selected_camera()
        return cam is not None

    def execute(self, context):
        """Compute a view synthesis for the current camera."""
        return run_view_synth(context.scene, op=self)



class ExportViewSynthesisOperator(bpy.types.Operator, ExportHelper):
    """An Operator to use the camera to render the NeRF Model, saving the output to the specified location"""

    bl_idname = "photogrammetry_importer.export_view_synthesis"
    bl_label = "Export View Synthesis as Image"
    bl_description = "Export camera properties to json and run the given script"

    # Hide the property by using a normal string instead of a string property
    filename_ext = ""

    @classmethod
    def poll(cls, context):
        """Return the availability status of the operator."""
        cam = get_selected_camera()
        return cam is not None

    def execute(self, context):
        """Compute a view synthesis for the current camera."""
        return run_view_synth(context.scene, save_to_dp=self.filepath, op=self)


class ExportViewSynthesisAnimOperator(bpy.types.Operator, ExportHelper):
    """An Operator to use the animation of the camera to render the NeRF Model"""

    bl_idname = "photogrammetry_importer.export_view_synthesis_anim"
    bl_label = "Export View Synthesis as Image Sequence"
    bl_description = "Export camera properties to json for each animation frame and run the given script"

    # Hide the property by using a normal string instead of a string property
    filename_ext = ""

    @classmethod
    def poll(cls, context):
        """Return the availability status of the operator."""
        cam = get_selected_camera()
        return cam is not None  # and cam.animation_data is not None

    def execute(self, context):
        """Compute a view synthesis for the current camera."""

        log_report(
            "INFO", "Export view synthesis for current camera with animation: ...", self
        )
        scene = context.scene

        command, temp_json_file, temp_array_file = create_instant_ngp_cmd(scene, self.filepath, op=self)

        animation_indices = get_scene_animation_indices()

        cameras = []
        for idx in animation_indices:
            bpy.context.scene.frame_set(idx)
            camera_relative_to_anchor, centroid_shift = shift_selected_camera_relative_to_anchor(scene)
            cameras.append(camera_relative_to_anchor)

        # Call before executing the child process
        InstantNGPFileHandler.write_instant_ngp_file(
            temp_json_file.name,
            cameras,
            ref_centroid_shift=centroid_shift,
        )

        child_process = subprocess.Popen(command)
        child_process.communicate()

        cleanup_tmp_files(temp_json_file, temp_array_file)

        log_report(
            "INFO", "Export view synthesis for current camera with Animation: Done", self
        )
        return {"FINISHED"}


def cleanup_tmp_files(temp_json_file, temp_array_file):
    if sys.platform == "win32":
        # Required for windows (https://docs.python.org/3.9/library/tempfile.html)
        temp_json_file.close()
        temp_array_file.close()
        os.unlink(temp_json_file.name)
        os.unlink(temp_array_file.name)

def show_image_in_blender(temp_array_file, camera_obj):
    # Call after executing the child process
    img_np_array = read_np_array_from_file(
        temp_array_file.name, use_pickle=False
    )

    blender_image = bpy.data.images.new(
        "view_synthesis_result",
        width=img_np_array.shape[1],
        height=img_np_array.shape[0],
    )
    img_np_array_flipped = np.flipud(img_np_array)
    blender_image.pixels = img_np_array_flipped.ravel()
    load_background_image(blender_image, camera_obj.name)

def shift_selected_camera_relative_to_anchor(scene):
    anchor_obj = bpy.data.objects[
        scene.view_synthesis_panel_settings.rotation_anchor_obj_name
    ]
    anchor_matrix_world = invert_transformation_matrix(
        np.array(anchor_obj.matrix_world)
    )
    # if the anchor obj was shifted to the center during import
    # apply the reverse translation so that the camera is relative to the original coordinate system
    centroid_shift = anchor_obj.get("centroid_shift", None)
    if centroid_shift is not None:
        anchor_matrix_world[0, 3] += centroid_shift[0]
        anchor_matrix_world[1, 3] += centroid_shift[1]
        anchor_matrix_world[2, 3] += centroid_shift[2]

    anchor_matrix_world_inverse = Matrix(anchor_matrix_world)

    camera_obj = get_selected_camera()
    camera_obj_relative_to_anchor = camera_obj.copy()
    camera_obj_relative_to_anchor.matrix_world = (
            anchor_matrix_world_inverse
            @ camera_obj_relative_to_anchor.matrix_world
    )

    camera_relative_to_anchor = get_computer_vision_camera(
        camera_obj_relative_to_anchor,
        camera_obj_relative_to_anchor.name,
        check_scale=False,
    )
    return camera_relative_to_anchor, centroid_shift


def create_instant_ngp_cmd(scene, output_dp, op=None):
    if sys.platform == "linux":
        temp_json_file = NamedTemporaryFile()
        temp_array_file = NamedTemporaryFile()
    elif sys.platform == "win32":
        temp_json_file = NamedTemporaryFile(delete=False)
        temp_array_file = NamedTemporaryFile(delete=False)
        # Required for windows (https://docs.python.org/3.9/library/tempfile.html)
        #  Whether the name can be used to open the file a second time, while the named temporary file is still open,
        #  varies across platforms (it can be so used on Unix; it cannot on Windows)
        temp_json_file.close()
        temp_array_file.close()
    else:
        assert False

    if (
            scene.view_synthesis_panel_settings.execution_environment
            == "CONDA"
    ):
        conda_exe_fp = scene.view_synthesis_panel_settings.conda_exe_fp
        conda_env_name = scene.view_synthesis_panel_settings.conda_env_name
        python_exe_fp = None
    elif (
            scene.view_synthesis_panel_settings.execution_environment
            == "DEFAULT PYTHON"
    ):
        python_exe_fp = scene.view_synthesis_panel_settings.python_exe_fp
        conda_exe_fp = None
        conda_env_name = None

    view_synthesis_exe_or_script_fp = (
        scene.view_synthesis_panel_settings.view_synthesis_executable_fp
    )
    view_synthesis_snapshot_fp = (
        scene.view_synthesis_panel_settings.view_synthesis_snapshot_fp
    )
    additional_system_dps = (
        scene.view_synthesis_panel_settings.additional_system_dps
    )
    samples_per_pixel = (
        scene.view_synthesis_panel_settings.samples_per_pixel
    )

    parameter_list = ["--load_snapshot", view_synthesis_snapshot_fp]
    parameter_list += ["--temp_json_ifp", temp_json_file.name]
    parameter_list += ["--temp_array_ofp", temp_array_file.name]
    parameter_list += ["--samples_per_pixel", str(samples_per_pixel)]
    if additional_system_dps.strip() != "":
        parameter_list += [
            "--additional_system_dps",
            additional_system_dps,
        ]
    if output_dp is not None and output_dp.strip() != "":
        parameter_list += [
            "--additional_output_dp",
            output_dp,
        ]

    assert os.path.isfile(view_synthesis_exe_or_script_fp)
    assert os.path.isfile(temp_json_file.name)
    assert os.path.isfile(temp_array_file.name)

    command = create_subprocess_command(
        view_synthesis_exe_or_script_fp,
        parameter_list,
        python_exe_fp=python_exe_fp,
        conda_exe_fp=conda_exe_fp,
        conda_env_name=conda_env_name,
    )
    cmd_call = " ".join(command)
    log_report("INFO", cmd_call, op)

    return command, temp_json_file, temp_array_file
