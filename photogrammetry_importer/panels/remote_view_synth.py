
def test():
    return "Test"


def run_view_synth_from_remote(panel_args, output_dp, camera_fp, op=None):
    InstantNGPFileHandler.write_instant_ngp_file(
        temp_json_file.name,
        [camera_relative_to_anchor],
        ref_centroid_shift=centroid_shift,
    )

    child_process = subprocess.Popen(command)
    child_process.communicate()

    show_image_in_blender(temp_array_file, get_selected_camera())

    cleanup_tmp_files(temp_json_file, temp_array_file)