import bpy
from bpy.props import (
    StringProperty,
    EnumProperty,
    IntProperty,
    PointerProperty,
    BoolProperty,
)
from photogrammetry_importer.panels.view_synthesis_operators import (
    RunViewSynthesisOperator,
    ExportViewSynthesisOperator,
    ExportViewSynthesisAnimOperator,
)
from photogrammetry_importer.blender_utility.retrieval_utility import (
    get_selected_camera,
)


class ViewSynthesisPanelSettings(bpy.types.PropertyGroup):
    """Class for the settings of the View Synthesis panel in the 3D view."""

    execution_environment_items = [
        ("CONDA", "Use Conda to run the view synthesis script", "", 1),
        (
            "DEFAULT PYTHON",
            "Use the standard Python environment to run the view synthesis script",
            "",
            2,
        ),
    ]
    execution_environment: EnumProperty(
        name="Execution Environment Type",
        description="Defines which environment is used to run the script",
        items=execution_environment_items,
    )
    conda_exe_fp: StringProperty(
        name="Conda Executable Name or File Path",
        description="",
        default="conda",
    )
    conda_env_name: StringProperty(
        name="Conda Environment Name",
        description="",
        default="base",
    )
    python_exe_fp: StringProperty(
        name="Python Executable Name or File Path",
        description="",
        default="python",
    )
    additional_system_dps: StringProperty(
        name="Additional System Paths to Run the Script",
        description="Additional system paths required to run the script. Two "
        "paths must be separated by a whitespace.",
        default="path/to/instant-ngp/build",
    )
    view_synthesis_executable_fp: StringProperty(
        name="View Synthesis Script File Name",
        description="",
        default="path/to/Blender-Addon-Photogrammetry-Importer/example_view_synthesis_scripts/instant_ngp.py",
    )
    view_synthesis_snapshot_fp: StringProperty(
        name="View Synthesis Output File Name",
        description="",
        default="path/to/instant-ngp/data/nerf/fox_colmap/snapshot.msgpack",
    )
    samples_per_pixel: IntProperty(
        name="Samples Per Pixel",
        description="",
        default=1,
    )
    rotation_anchor_obj_name: StringProperty(
        name="Rotation Anchor Object",
        description="The rotation of this object is considered for view "
        " synthesis. This allows to rotate the corresponding input scene.",
        default="OpenGL Point Cloud",
    )
    use_camera_keyframes_for_rendering: BoolProperty(
        name="Use Camera Keyframes",
        description="Use the Camera Keyframes instead of Animation Frames.",
        default=True,
    )
    render_solid_background: BoolProperty(
        name="Render solid Background",
        description="Use a solid background color for all areas outside of the learned scene boundaries",
        default=True,
    )
    render_semantic_color: BoolProperty(
        name="Render semantic false coloring",
        description="Render the semantic class color instead of the true RGB color",
        default=False,
    )
    cuda_device: IntProperty(
        name="Cuda Device",
        description="",
        default=0,
    )


class ViewSynthesisPanel(bpy.types.Panel):
    """Class that defines the view synthesis panel in the 3D view."""

    bl_label = "View Synthesis Panel"
    bl_idname = "IMPORT_VIEW_SYNTHESIS_PT_manage_view_synthesis_visualization"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PhotogrammetryImporter"

    @classmethod
    def poll(cls, context):
        """Return the availability status of the panel."""
        return True

    @classmethod
    def register(cls):
        """Register properties and operators corresponding to this panel."""

        bpy.utils.register_class(ViewSynthesisPanelSettings)
        bpy.types.Scene.view_synthesis_panel_settings = PointerProperty(
            type=ViewSynthesisPanelSettings
        )
        bpy.utils.register_class(RunViewSynthesisOperator)
        bpy.utils.register_class(ExportViewSynthesisOperator)
        bpy.utils.register_class(ExportViewSynthesisAnimOperator)

    @classmethod
    def unregister(cls):
        """Unregister properties and operators corresponding to this panel."""
        bpy.utils.unregister_class(ViewSynthesisPanelSettings)
        del bpy.types.Scene.view_synthesis_panel_settings
        bpy.utils.unregister_class(RunViewSynthesisOperator)
        bpy.utils.unregister_class(ExportViewSynthesisOperator)
        bpy.utils.unregister_class(ExportViewSynthesisAnimOperator)

    def draw(self, context):
        """Draw the panel with corrresponding properties and operators."""
        settings = context.scene.view_synthesis_panel_settings
        layout = self.layout
        view_synthesis_box = layout.box()
        selected_cam = get_selected_camera()

        row = view_synthesis_box.row()
        row.prop(settings, "execution_environment", text="Script Environment")
        if settings.execution_environment == "CONDA":
            row = view_synthesis_box.row()
            row.prop(
                settings, "conda_exe_fp", text="Conda Executable File Path"
            )
            row = view_synthesis_box.row()
            row.prop(settings, "conda_env_name", text="Conda Environment Name")
        elif settings.execution_environment == "DEFAULT PYTHON":
            row = view_synthesis_box.row()
            row.prop(
                settings,
                "python_exe_fp",
                text="Default Python Executable Name or File Path",
            )
        else:
            pass

        row = view_synthesis_box.row()
        row.prop(
            settings,
            "additional_system_dps",
            text="Additional System Paths",
        )
        row = view_synthesis_box.row()
        row.prop(
            settings,
            "view_synthesis_executable_fp",
            text="Script",
        )
        row = view_synthesis_box.row()
        row.prop(
            settings,
            "view_synthesis_snapshot_fp",
            text="Training Snapshot (Trained Model)",
        )
        row = view_synthesis_box.row()
        row.prop(
            settings,
            "cuda_device",
            text="Cuda Device",
        )
        row = view_synthesis_box.row()
        row.prop(settings, "samples_per_pixel", text="Samples Per Pixel")

        row = view_synthesis_box.row()
        row.prop(
            settings, "rotation_anchor_obj_name", text="Rotation Anchor Object"
        )
        row = view_synthesis_box.row()
        row.prop(
            settings, "render_solid_background", text="Render Solid Background"
        )
        row = view_synthesis_box.row()
        row.prop(
            settings, "render_semantic_color", text="Render Semantic Color"
        )

        view_synthesis_save_box = view_synthesis_box.box()
        view_synthesis_save_box.label(
            text="Run View Synthesis for current Camera"
        )
        row = view_synthesis_save_box.row()
        row.operator(RunViewSynthesisOperator.bl_idname)

        view_synthesis_export_box = view_synthesis_box.box()
        view_synthesis_export_box.label(
            text="Export View Synthesis for current Camera"
        )
        row = view_synthesis_export_box.row()
        row.operator(ExportViewSynthesisOperator.bl_idname)

        # Render Sequence Settings + Operator
        row = view_synthesis_export_box.row()
        row.prop(
            settings,
            "use_camera_keyframes_for_rendering",
            text="Use Camera Keyframes",
        )
        row.enabled = (
            selected_cam is not None
            and selected_cam.animation_data is not None
        )
        row = view_synthesis_export_box.row()
        row.operator(ExportViewSynthesisAnimOperator.bl_idname)
