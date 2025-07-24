import sys
import subprocess
import os
import numpy as np
import bpy
from mathutils import Matrix
from tempfile import NamedTemporaryFile
from bpy_extras.io_utils import ExportHelper
from datetime import datetime


from photogrammetry_importer.utility.np_utility import (
    invert_transformation_matrix,
)
from photogrammetry_importer.blender_utility.retrieval_utility import (
    get_selected_camera,
    get_scene_animation_indices,
    get_object_animation_indices,
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
    log_report("INFO", "Compute view synthesis for current camera: ...", op)

    panel_args = extract_args_from_scene(scene)
    command, temp_json_file, temp_array_file = create_instant_ngp_cmd(
        panel_args, output_dp=save_to_dp, op=op
    )

    camera_relative_to_anchor, centroid_shift = (
        shift_selected_camera_relative_to_anchor(scene)
    )

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

    log_report("INFO", "Compute view synthesis for current camera: Done", op)
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
        # return run_view_synth(context.scene, op=self)
        args = extract_args_from_scene(context.scene)
        camera_relative_to_anchor, centroid_shift = (
            shift_selected_camera_relative_to_anchor(context.scene)
        )
        return start_view_synth_on_remote(args, [camera_relative_to_anchor], centroid_shift, op=self)


class ExportViewSynthesisOperator(bpy.types.Operator, ExportHelper):
    """An Operator to use the camera to render the NeRF Model, saving the output to the specified location"""

    bl_idname = "photogrammetry_importer.export_view_synthesis"
    bl_label = "Export View Synthesis as Image"
    bl_description = (
        "Export camera properties to json and run the given script"
    )

    # Hide the property by using a normal string instead of a string property
    filename_ext = ""

    @classmethod
    def poll(cls, context):
        """Return the availability status of the operator."""
        cam = get_selected_camera()
        return cam is not None

    def execute(self, context):
        """Compute a view synthesis for the current camera."""
        # return run_view_synth(context.scene, save_to_dp=self.filepath, op=self)
        args = extract_args_from_scene(context.scene)

        camera_relative_to_anchor, centroid_shift = (
            shift_selected_camera_relative_to_anchor(context.scene)
        )

        return start_view_synth_on_remote(args, [camera_relative_to_anchor], centroid_shift, save_to_dp=self.filepath, op=self)


class ExportViewSynthesisAnimOperator(bpy.types.Operator):
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
            "INFO",
            "Export view synthesis for current camera with animation: ...",
            self,
        )
        scene = context.scene

        args = extract_args_from_scene(scene)
       
        use_camera_keyframes = (
            scene.view_synthesis_panel_settings.use_camera_keyframes_for_rendering
        )
        selected_cam = get_selected_camera()
        if (
            use_camera_keyframes
            and selected_cam is not None
            and selected_cam.animation_data is not None
        ):
            animation_indices = get_object_animation_indices(selected_cam)
        else:
            animation_indices = get_scene_animation_indices()

        cameras = []
        for idx in animation_indices:
            bpy.context.scene.frame_set(idx)
            camera_relative_to_anchor, centroid_shift = (
                shift_selected_camera_relative_to_anchor(scene)
            )
            cameras.append(camera_relative_to_anchor)

        start_view_synth_on_remote(args, cameras, centroid_shift, op=self)

        log_report(
            "INFO",
            "Export view synthesis for current camera with Animation: Started on Remote",
            self,
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


def extract_args_from_scene(scene, op=None):
    args = {}
    if scene.view_synthesis_panel_settings.execution_environment == "CONDA":
        args["conda_exe_fp"] = scene.view_synthesis_panel_settings.conda_exe_fp
        args["conda_env_name"] = scene.view_synthesis_panel_settings.conda_env_name
        args["python_exe_fp"] = None
    elif (
        scene.view_synthesis_panel_settings.execution_environment
        == "DEFAULT PYTHON"
    ):
        args["python_exe_fp"] = scene.view_synthesis_panel_settings.python_exe_fp
        args["conda_exe_fp"] = None
        args["conda_env_name"] = None

    args["view_synthesis_exe_or_script_fp"] = (
        scene.view_synthesis_panel_settings.view_synthesis_executable_fp
    )
    args["view_synthesis_snapshot_fp"] = (
        scene.view_synthesis_panel_settings.view_synthesis_snapshot_fp
    )
    args["additional_system_dps"] = (
        scene.view_synthesis_panel_settings.additional_system_dps
    )
    args["samples_per_pixel"] = scene.view_synthesis_panel_settings.samples_per_pixel
    args["render_solid_background"] = (
        scene.view_synthesis_panel_settings.render_solid_background
    )
    args["render_semantic_color"] = (
        scene.view_synthesis_panel_settings.render_semantic_color
    )
    args["cuda_device"] = scene.view_synthesis_panel_settings.cuda_device
    
    return args


def create_instant_ngp_cmd(args, output_dp, op=None):
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

    parameter_list = ["--load_snapshot", args["view_synthesis_snapshot_fp"]]
    parameter_list += ["--temp_json_ifp", temp_json_file.name]
    parameter_list += ["--temp_array_ofp", temp_array_file.name]
    parameter_list += ["--samples_per_pixel", str(args["samples_per_pixel"])]
    if args["render_solid_background"]:
        parameter_list += ["--render_solid_background"]
    if args["render_semantic_color"]:
        parameter_list += ["--render_semantic_color"]
    parameter_list += ["--cuda_device", str(args["cuda_device"])]
    if args["additional_system_dps"].strip() != "":
        parameter_list += [
            "--additional_system_dps",
            "\"" + args["additional_system_dps"] + "\"",
        ]
    if output_dp is not None and output_dp.strip() != "":
        parameter_list += [
            "--additional_output_dp",
            output_dp,
        ]
    # assert os.path.isfile(args["view_synthesis_exe_or_script_fp"])
    # assert os.path.isfile(temp_json_file.name)
    # assert os.path.isfile(temp_array_file.name)

    command = create_subprocess_command(
        args["view_synthesis_exe_or_script_fp"],
        parameter_list,
        python_exe_fp=args["python_exe_fp"],
        conda_exe_fp=args["conda_exe_fp"],
        conda_env_name=args["conda_env_name"],
    )
    cmd_call = " ".join(command)
    log_report("INFO", cmd_call, op)

    return command, temp_json_file, temp_array_file



def start_view_synth_on_remote(panel_args, cameras, centroid_shift, save_to_dp=None, op=None):
    log_report("INFO", "Compute view synthesis for current camera: ...", op)

    import spur
    import shutil
    import json
    
    # get creds
    with open("c:\\Users\\val60188\\Documents\\Blender\\creds.json", "r") as fp:
        creds = json.load(fp)
    
    # connect to remote server
    shell = spur.SshShell(
        hostname="10.21.1.227",
        username=creds["username"],
        password=creds["pwd"],
        # shell_type=spur.ssh.ShellTypes.minimal
    )

    remote_base_dp = "/mnt/DATA3-2TB/val60188/blender"
    cwd = "/mnt/DATA3-2TB/val60188/blender/Blender-Addon-Photogrammetry-Importer"

    remote_output_dp = None
    # if rendering video, store outputs on server in a permanent location
    video_rendering = len(cameras) > 1
    if video_rendering:
        remote_output_dp = f"{remote_base_dp}/output"

    command, temp_json_file, temp_array_file = create_instant_ngp_cmd(
        panel_args, output_dp=remote_output_dp, op=op
    )

    # Call before executing the child process
    InstantNGPFileHandler.write_instant_ngp_file(
        temp_json_file.name,
        cameras,
        ref_centroid_shift=centroid_shift,
    )

    # upload files to server
    json_fp = f"{remote_base_dp}/tmp/json"
    img_dp = f"{remote_base_dp}/tmp/images"

    with shell.open(json_fp, "wtb") as remote_file:
        with open(temp_json_file.name, "rb") as local_file:
            shutil.copyfileobj(local_file, remote_file)

    # make sure temp directory exists and is empty (by deleting first if it already exists)
    shell.run(["bash", "-c", f'[ -d "{img_dp}" ] && rm -r "{img_dp}"'])
    shell.run(["mkdir", "-p", img_dp])

    cmd_call = " ".join(command).replace(temp_json_file.name, json_fp).replace(temp_array_file.name, img_dp)
    args = {
        "json_fp": json_fp,
        "img_fp": img_dp,
        "cmd": cmd_call
    }

    args_str = []
    for key, value in args.items():
        args_str.append(f"{key}='{value}'") 
    args_str = ",".join(args_str)


    print("cmd_call:", cmd_call)

    cmd_list = ["/home/val60188/miniconda3/bin/conda", "run", "-n", "rs", 
                        "python", "-c", f"from remote_view_synth import run; run({args_str})"]
    if video_rendering:
        # run view synth on remote, dont wait for result
        result = shell.spawn(cmd_list, 
                        cwd=cwd)
    else:
        # run view synth on remote, wait for result
        result = shell.run(cmd_list, 
                        cwd=cwd)
    
        # print("Err Code", result.return_code)
        out = result.output.decode('utf-8').replace("\\n", "\n")
        errs = result.stderr_output.decode('utf-8').replace("\\n", "\n")
        print("Returned: ", out)
        print("Errs: ", errs)

    # copy extracted image back
    if not video_rendering:
        # get number of images files created by view_synth
        img_files = shell.run(["ls", img_dp]).output.decode('utf-8').splitlines()
        timestr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        for index, img_name in enumerate(img_files):
            img_fp = img_dp + "/" + img_name  # since on linux remote
            
            if index == 0:
                with shell.open(img_fp, "rb") as remote_file:
                    with open(temp_array_file.name, "wb") as local_file:
                        shutil.copyfileobj(remote_file, local_file)
            
            if save_to_dp is not None:
                # copy to save_to_dp as .npy
                tmp_local = os.path.join(save_to_dp, "tmp_array.npy")
                with shell.open(img_fp, "rb") as remote_file:
                    with open(tmp_local, "wb") as local_file:
                        shutil.copyfileobj(remote_file, local_file)
                
                # rewrite as image
                img_np_array = read_np_array_from_file(
                    tmp_local, use_pickle=False
                )
                img_np_array = (img_np_array * 255.0).astype(np.uint8)
                ofp = os.path.join(save_to_dp, timestr + f"_{index}.png")
                os.makedirs(os.path.dirname(ofp), exist_ok=True )
                from PIL import Image

                img = Image.fromarray(img_np_array)
                img.save(ofp)
                os.remove(tmp_local)
        

        # report results for last saved image
        log_report("INFO", f"Saved image to {ofp}", op)
        
        # show image in blender, then delete temp files
        show_image_in_blender(temp_array_file, get_selected_camera())
        cleanup_tmp_files(temp_json_file, temp_array_file)
    
    log_report("INFO", "Compute view synthesis for current camera: Done", op)
    return {"FINISHED"}


