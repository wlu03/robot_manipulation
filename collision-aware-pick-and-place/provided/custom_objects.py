"""Register bundled object meshes when they are absent from the AMS registry."""

from __future__ import annotations

import copy
import pathlib
from collections.abc import Mapping, MutableMapping

ASSET_DIR = pathlib.Path(__file__).resolve().parents[1] / "assets"
CUSTOM_OBJECT_DIRS = {
    "tall_can": ASSET_DIR / "tall_can",
    "tall_box": ASSET_DIR / "tall_box",
    "mustard_bottle": ASSET_DIR / "mustard_bottle",
}


def _replace_mesh_value(value, mesh_path: pathlib.Path):
    # AMS uses Path.parent to locate the object's collision_mesh directory.
    if isinstance(value, tuple):
        return (mesh_path,)
    if isinstance(value, list):
        return [mesh_path]
    return mesh_path


def install_custom_objects() -> None:
    """Register missing MP2 fixtures while preserving objects supplied by AMS."""

    from ams.utilities.scene_builder import SceneBuilder

    if getattr(SceneBuilder, "_mp2_custom_objects_installed", False):
        return

    original_init = SceneBuilder.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        registry = getattr(self, "registry", None)
        if not isinstance(registry, MutableMapping) or not registry:
            raise RuntimeError("AMS SceneBuilder registry is unavailable")

        # Copy AMS metadata from an existing object; replace only the mesh paths.
        template_name = "pear" if "pear" in registry else next(iter(registry))
        template = registry[template_name]
        if not isinstance(template, Mapping):
            raise RuntimeError("unexpected AMS SceneBuilder registry schema")

        for name, object_dir in CUSTOM_OBJECT_DIRS.items():
            if name in registry:
                continue
            visual_path = object_dir / "visual.obj"
            collision_dir = object_dir / "collision_mesh"
            if not visual_path.is_file():
                raise RuntimeError(f"missing MP2 custom visual mesh: {visual_path}")
            if not collision_dir.is_dir() or not any(collision_dir.glob("*.obj")):
                raise RuntimeError(
                    f"missing MP2 custom collision meshes: {collision_dir}"
                )

            entry = copy.deepcopy(dict(template))

            if "visual" not in entry:
                raise RuntimeError("AMS object registry entry has no 'visual' mesh")
            entry["visual"] = _replace_mesh_value(entry["visual"], visual_path)

            texture_path = object_dir / "texture_map.png"
            if texture_path.is_file():
                entry["texture"] = _replace_mesh_value(
                    entry.get("texture"), texture_path
                )

            collision_keys = [key for key in entry if "collision" in str(key).lower()]
            for key in collision_keys:
                # SceneBuilder uses this Path's parent to find collision_mesh/.
                entry[key] = _replace_mesh_value(entry[key], visual_path)

            if not collision_keys:
                entry["collision"] = visual_path

            registry[name] = entry

    SceneBuilder.__init__ = patched_init
    SceneBuilder._mp2_custom_objects_installed = True
