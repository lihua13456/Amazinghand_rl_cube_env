"""Create an Isaac Lab friendly copy of the existing collision hand USD."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_USD = HERE.parent / "usd_collision_grasp" / "AH_Right_collision.usd"
TRAINING_USD = HERE / "assets" / "AH_Right_rl.usd"


def prepare_robot_usd() -> Path:
    """Remove the embedded scene and normalize unlimited ball-joint axis metadata.

    Isaac Lab owns the physics scene. The source asset also contains one under
    /AH_Right; cloning that scene for every environment is invalid. Isaac Sim's
    tensor articulation interface also expects spherical joints to use axis X.
    These joints have no cone limits, so the axis attribute does not constrain
    their physical rotation. The original USD is never edited.
    """
    from pxr import Usd, UsdPhysics

    if not SOURCE_USD.is_file():
        raise FileNotFoundError(f"Collision robot USD not found: {SOURCE_USD}")
    stage = Usd.Stage.Open(str(SOURCE_USD))
    if stage is None or not stage.GetDefaultPrim():
        raise RuntimeError(f"Cannot open robot USD: {SOURCE_USD}")
    scene = stage.GetPrimAtPath("/AH_Right/PhysicsScene")
    if not scene or not scene.IsA(UsdPhysics.Scene):
        raise RuntimeError("Expected /AH_Right/PhysicsScene in collision robot USD")
    stage.RemovePrim(scene.GetPath())
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.SphericalJoint):
            joint = UsdPhysics.SphericalJoint(prim)
            if joint.GetConeAngle0LimitAttr().Get() != -1.0 or joint.GetConeAngle1LimitAttr().Get() != -1.0:
                raise RuntimeError(f"Cannot change axis of limited spherical joint: {prim.GetPath()}")
            joint.GetAxisAttr().Set("X")
    TRAINING_USD.parent.mkdir(parents=True, exist_ok=True)
    if not stage.Export(str(TRAINING_USD)):
        raise RuntimeError(f"Cannot export training robot USD: {TRAINING_USD}")
    return TRAINING_USD
