# Asset pipeline: Gazebo URDF -> Isaac Sim USD

The robot model is the **same physical UR5e + Robotiq-140** the Gazebo project
(`remoteservertraining-UR5e`) uses -- not a stand-in. It comes from that repo's own
`ur_description`/`robotiq_2f_gripper_description` xacro, expanded with `ur_type:=ur5e`,
so link lengths, meshes, and joint limits match exactly.

Checked into this repo:
- `urdf/ur5e_robotiq140.urdf` -- raw xacro-expanded output (`ur_type:=ur5e use_gazebo:=false use_camera:=false`).
- `urdf/ur5e_robotiq140.isaac.urdf` -- patched for Isaac Sim (`patch_urdf.py`): `package://` refs resolved to real paths, `.dae` visual meshes swapped for `.stl` collision meshes (Isaac Sim's importer crashes on these particular Collada files).
- `meshes/` -- copied out of the Gazebo repo's `ur_description`/`robotiq_2f_gripper_description` packages, so this repo doesn't depend on that one being checked out alongside it.

**Not** checked in: the converted `.usd`. It's a generated build artifact (same
reasoning the Gazebo repo's own `.gitignore` uses for `build/`/`install/`/`log/`) --
regenerate it with `convert_urdf_to_usd.py` rather than committing a multi-MB binary
that's fully reproducible from the URDF+meshes above.

## Regenerating the USD

```
# Windows
C:\isaacsim\python.bat convert_urdf_to_usd.py

# Linux (Isaac Sim pip install or vast.ai)
python convert_urdf_to_usd.py
```

Output goes to `%USERPROFILE%\isaacsim_assets\isaacsimtraining\usd\ur5e_robotiq140.usd`
(Windows) or `~/.cache/isaacsimtraining/usd/ur5e_robotiq140.usd` (Linux) by default;
override with `--dest-path`, or set `UR5E_USD_PATH` (read by `ur5e_grasp_env.py`) to
point at wherever you put it.

## Regenerating the URDF itself (only needed if the Gazebo repo's robot description changes)

Requires ROS 2 Humble + xacro (this repo's own patch/convert scripts don't need ROS,
only this one step does):

```bash
cd remoteservertraining-UR5e
colcon build --packages-select ur_description robotiq_2f_gripper_description
source install/setup.bash
xacro src/ur_description/urdf/ur_robotiq_140.urdf.xacro \
  ur_type:=ur5e use_gazebo:=false use_camera:=false name:=ur5e \
  -o /path/to/isaacsimtraining/assets/urdf/ur5e_robotiq140.urdf
# then copy src/ur_description/meshes/ur5e -> assets/meshes/ur_description/meshes/ur5e
# and src/robotiq_2f_gripper_ros2/robotiq_2f_gripper_description/meshes -> assets/meshes/robotiq_2f_gripper_description/meshes
python3 patch_urdf.py
```

`use_camera:=false`: the wrist camera (used by the Gazebo project's perception/YOLO
target-source mode) isn't wired into this Isaac Sim port yet -- this env is
ground-truth-target only for now. Re-run with `use_camera:=true` and extend
`ur5e_grasp_env.py` if/when perception mode gets ported too.

## Regenerating on Windows when your checkout lives under WSL

Isaac Sim's URDF importer (confirmed live, Isaac Sim 5.1) fails to resolve mesh
references through a `\\wsl.localhost\...` UNC path. If this repo is checked out in
WSL, copy `assets/urdf/` and `assets/meshes/` to a native Windows folder first, patch
the copy's mesh paths to point at that Windows folder (`patch_urdf.py --meshes-root
C:\path\to\copy\meshes`), then run `convert_urdf_to_usd.py --urdf-path
C:\path\to\copy\urdf\ur5e_robotiq140.isaac.urdf` from there.
