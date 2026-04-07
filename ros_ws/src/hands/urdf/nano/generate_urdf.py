"""
NanoHand 10-DOF URDF Generator with Mimic Joints

Chain per finger:
  palm → [wiggle @ MCP] → base(invis) → [curl_mcp] → knuckle(mesh)
       → [curl_pip, mimic] → mid(mesh) → [curl_dip, mimic] → tip(mesh)

Still 10 DOF: wiggle + curl_mcp per finger.
PIP and DIP follow MCP via mimic with measured multipliers.

IMPORTANT: After generating, update the mimic multipliers with
physical measurements from the real hand using a protractor!
  distal_angle = multiplier * proximal_angle + offset
"""
import trimesh
import numpy as np
import os
import sys
import shutil

mesh_dir = "meshes"
if not os.path.exists(mesh_dir):
    print(f"ERROR: {mesh_dir}/ folder not found")
    sys.exit(1)

# ── Step 1: Rename files to remove spaces ──────────────────────────
print("=== Renaming mesh files (removing spaces) ===")
for f in os.listdir(mesh_dir):
    if ' ' in f:
        new_name = f.replace(' ', '_')
        old_path = os.path.join(mesh_dir, f)
        new_path = os.path.join(mesh_dir, new_name)
        if not os.path.exists(new_path):
            shutil.move(old_path, new_path)
            print(f"  Renamed: {f} → {new_name}")

# ── Step 2: Load meshes ───────────────────────────────────────────
print("\n=== Loading meshes ===")

# Expected naming: Knuckle_FINGER_section.stl, Mid_FINGER_section.stl, Tip_FINGER_section.stl
fingers = ['pinky', 'ring', 'middle', 'index', 'thumb']
finger_meshes = {}  # finger_meshes['pinky'] = {'knuckle': mesh, 'mid': mesh, 'tip': mesh}

# Load palm
palm_path = os.path.join(mesh_dir, 'palm.stl')
if not os.path.exists(palm_path):
    print("ERROR: palm.stl not found")
    sys.exit(1)
palm = trimesh.load(palm_path)
palm_center = palm.centroid
palm_bounds = palm.bounds
palm_size = palm_bounds[1] - palm_bounds[0]
print(f"  palm: centroid=[{palm_center[0]:.4f}, {palm_center[1]:.4f}, {palm_center[2]:.4f}]")

# Load finger segments
for fn in fingers:
    finger_meshes[fn] = {}
    for seg in ['Knuckle', 'Mid', 'Tip']:
        # Try various naming patterns (finger first or segment first)
        candidates = [
            f"{fn.capitalize()}_{seg.lower()}_section.stl",
            f"{seg}_{fn}_section.stl",
            f"{seg}_{fn.capitalize()}_section.stl",
            f"{seg}_{fn}_Section.stl",
            f"{fn}_{seg.lower()}_section.stl",
        ]
        found = False
        for cand in candidates:
            path = os.path.join(mesh_dir, cand)
            if os.path.exists(path):
                finger_meshes[fn][seg.lower()] = trimesh.load(path)
                s = finger_meshes[fn][seg.lower()].bounding_box.extents
                print(f"  {fn} {seg}: {s[0]:.3f} x {s[1]:.3f} x {s[2]:.3f}")
                found = True
                break
        if not found:
            # Try listing files for this finger/segment combo using word boundaries
            for f in sorted(os.listdir(mesh_dir)):
                fl = f.lower()
                if not fl.endswith('.stl'):
                    continue
                parts = fl.replace('.stl', '').split('_')
                if seg.lower() in parts and fn.lower() in parts:
                    finger_meshes[fn][seg.lower()] = trimesh.load(os.path.join(mesh_dir, f))
                    s = finger_meshes[fn][seg.lower()].bounding_box.extents
                    print(f"  {fn} {seg}: {s[0]:.3f} x {s[1]:.3f} x {s[2]:.3f} (from {f})")
                    found = True
                    break
        if not found:
            print(f"  WARNING: {fn} {seg} not found!")

# ── Step 3: Compute palm geometry ──────────────────────────────────
thin_idx = np.argmin(palm_size)
palm_normal = np.zeros(3)
palm_normal[thin_idx] = 1.0

# Finger extend direction from non-thumb finger centroids
non_thumb_centroids = []
for fn in ['pinky', 'ring', 'middle', 'index']:
    if 'knuckle' in finger_meshes.get(fn, {}):
        non_thumb_centroids.append(finger_meshes[fn]['knuckle'].centroid)

if non_thumb_centroids:
    avg_finger = np.mean(non_thumb_centroids, axis=0)
    finger_extend_dir = avg_finger - palm_center
    finger_extend_dir[thin_idx] = 0
    if np.linalg.norm(finger_extend_dir) > 0.001:
        finger_extend_dir = finger_extend_dir / np.linalg.norm(finger_extend_dir)
else:
    finger_extend_dir = np.array([1, 0, 0])

across_palm = np.cross(finger_extend_dir, palm_normal)
if np.linalg.norm(across_palm) > 0.001:
    across_palm = across_palm / np.linalg.norm(across_palm)

print(f"\nPalm normal: {palm_normal}")
print(f"Finger extend: {finger_extend_dir}")
print(f"Across palm: {across_palm}")

# ── Step 4: Compute joint positions per finger ─────────────────────
print("\n=== Computing joint positions ===")

joint_positions = {}  # joint_positions['pinky'] = {'mcp': pos, 'pip': pos, 'dip': pos}
finger_dirs = {}
curl_axes = {}

for fn in fingers:
    segs = finger_meshes.get(fn, {})
    if 'knuckle' not in segs or 'mid' not in segs or 'tip' not in segs:
        print(f"  SKIPPING {fn} - missing segments")
        continue

    knuckle = segs['knuckle']
    mid = segs['mid']
    tip = segs['tip']

    # MCP joint = where knuckle is closest to palm
    dists = np.linalg.norm(knuckle.vertices - palm_center, axis=1)
    thresh = np.percentile(dists, 10)
    mcp_pos = knuckle.vertices[dists <= thresh].mean(axis=0)

    # PIP joint = where mid segment is closest to knuckle centroid
    knuckle_center = knuckle.centroid
    dists = np.linalg.norm(mid.vertices - knuckle_center, axis=1)
    thresh = np.percentile(dists, 10)
    pip_pos = mid.vertices[dists <= thresh].mean(axis=0)

    # DIP joint = where tip segment is closest to mid centroid
    mid_center = mid.centroid
    dists = np.linalg.norm(tip.vertices - mid_center, axis=1)
    thresh = np.percentile(dists, 10)
    dip_pos = tip.vertices[dists <= thresh].mean(axis=0)

    joint_positions[fn] = {'mcp': mcp_pos, 'pip': pip_pos, 'dip': dip_pos}

    # Finger direction: MCP to tip centroid
    tip_center = tip.centroid
    fd = tip_center - mcp_pos
    if np.linalg.norm(fd) > 0.001:
        fd = fd / np.linalg.norm(fd)
    finger_dirs[fn] = fd

    # Curl axis: perpendicular to finger direction and palm normal
    ca = np.cross(fd, palm_normal)
    if np.linalg.norm(ca) < 0.01:
        ca = across_palm.copy()
    else:
        ca = ca / np.linalg.norm(ca)
    curl_axes[fn] = ca

    print(f"\n  {fn}:")
    print(f"    MCP: [{mcp_pos[0]:.5f}, {mcp_pos[1]:.5f}, {mcp_pos[2]:.5f}]")
    print(f"    PIP: [{pip_pos[0]:.5f}, {pip_pos[1]:.5f}, {pip_pos[2]:.5f}]")
    print(f"    DIP: [{dip_pos[0]:.5f}, {dip_pos[1]:.5f}, {dip_pos[2]:.5f}]")
    print(f"    dir: [{fd[0]:.3f}, {fd[1]:.3f}, {fd[2]:.3f}]")
    print(f"    curl axis: [{ca[0]:.3f}, {ca[1]:.3f}, {ca[2]:.3f}]")

# ── Step 5: Find mesh filenames ────────────────────────────────────
def find_mesh_file(segment, finger):
    """Find the actual filename in meshes/ for a given segment and finger."""
    # After renaming, files are like: Pinky_knuckle_section.stl or Knuckle_pinky_section.stl
    for f in sorted(os.listdir(mesh_dir)):
        fl = f.lower()
        if not fl.endswith('.stl'):
            continue
        parts = fl.replace('.stl', '').split('_')
        # Check if both finger name and segment name appear as separate words
        if finger.lower() in parts and segment.lower() in parts:
            return f
    return None

# ── Step 6: Build URDF ────────────────────────────────────────────
print("\n=== Generating URDF with mimic joints ===")

# MIMIC MULTIPLIERS - UPDATE THESE WITH PHYSICAL MEASUREMENTS!
# Format: (pip_multiplier, pip_offset, dip_multiplier, dip_offset)
# Measure with protractor: curl finger using servo GUI at 2-3 angles
# pip_angle = pip_multiplier * mcp_angle + pip_offset
# dip_angle = dip_multiplier * mcp_angle + dip_offset
MIMIC_VALUES = {
    'pinky':  (2.92, 0.0, 3.94, 0.0),
    'ring':   (2.31, 0.0, 2.85, 0.0),
    'middle': (2.20, 0.0, 2.85, 0.0),
    'index':  (0.7, 0.0, 0.5, 0.0),   # PLACEHOLDER - hardware issue, fix later
    'thumb':  (0.71, 0.0, 0.54, 0.0),
}

# Servo configs: (name, wiggle_limits, curl_limits, curl_reversed)
configs = [
    ("pinky",  (-0.256, 0.256), (0.0, 1.57), True),
    ("ring",   (-0.256, 0.256), (0.0, 1.57), True),
    ("middle", (-0.256, 0.256), (0.0, 1.57), True),
    ("index",  (-0.256, 0.256), (0.0, 1.57), True),
    ("thumb",  (-1.3, 1.3),     (0.0, 1.57), False),
]

lines = [
    '<?xml version="1.0"?>',
    '<robot name="nanohand">',
    '',
    '  <!-- NanoHand 10-DOF URDF with Mimic Joints -->',
    '  <!-- 3 curl joints per finger: MCP (controlled) + PIP (mimic) + DIP (mimic) -->',
    '  <!-- UPDATE mimic multipliers after physical measurement with protractor! -->',
    '',
    '  <!-- ===== PALM ===== -->',
    '  <link name="palm">',
    '    <visual>',
    '      <origin xyz="0 0 0" rpy="0 0 0"/>',
    '      <geometry><mesh filename="package://hands/urdf/nano/meshes/palm.stl"/></geometry>',
    '      <material name="palm_mat"><color rgba="0.9 0.75 0.2 1.0"/></material>',
    '    </visual>',
    '    <collision>',
    '      <origin xyz="0 0 0" rpy="0 0 0"/>',
    '      <geometry><mesh filename="package://hands/urdf/nano/meshes/palm.stl"/></geometry>',
    '    </collision>',
    '    <inertial>',
    '      <mass value="0.15"/>',
    '      <origin xyz="0 0 0"/>',
    '      <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/>',
    '    </inertial>',
    '  </link>',
]

for fn, wig_lim, curl_lim, reversed_curl in configs:
    if fn not in joint_positions:
        print(f"  SKIPPING {fn}")
        continue

    jp = joint_positions[fn]
    fd = finger_dirs[fn]
    ca = curl_axes[fn].copy()

    if reversed_curl:
        ca = -ca

    # Wiggle axis = palm normal
    wa = palm_normal.copy()

    # Joint positions in assembly coords
    p_mcp = jp['mcp']
    p_pip = jp['pip']
    p_dip = jp['dip']

    # Relative joint positions (each relative to parent frame)
    pip_rel = p_pip - p_mcp  # PIP relative to MCP frame
    dip_rel = p_dip - p_pip  # DIP relative to PIP frame

    # Mesh offsets (negative of joint position in assembly coords)
    knuckle_offset = -p_mcp
    mid_offset = -p_pip
    tip_offset = -p_dip

    # Find mesh filenames
    knuckle_file = find_mesh_file('knuckle', fn)
    mid_file = find_mesh_file('mid', fn)
    tip_file = find_mesh_file('tip', fn)

    if not all([knuckle_file, mid_file, tip_file]):
        print(f"  SKIPPING {fn} - missing mesh files")
        continue

    # Mimic values
    pip_mult, pip_off, dip_mult, dip_off = MIMIC_VALUES[fn]

    print(f"  {fn}: knuckle={knuckle_file}, mid={mid_file}, tip={tip_file}")

    lines.extend([
        f'',
        f'  <!-- ===== {fn.upper()} (wiggle + 3 curl joints) ===== -->',
        f'  <!-- Wiggle joint at MCP -->',
        f'  <joint name="{fn}_wiggle" type="revolute">',
        f'    <parent link="palm"/>',
        f'    <child link="{fn}_base"/>',
        f'    <origin xyz="{p_mcp[0]:.6f} {p_mcp[1]:.6f} {p_mcp[2]:.6f}" rpy="0 0 0"/>',
        f'    <axis xyz="{wa[0]:.4f} {wa[1]:.4f} {wa[2]:.4f}"/>',
        f'    <limit lower="{wig_lim[0]}" upper="{wig_lim[1]}" effort="1.0" velocity="3.0"/>',
        f'  </joint>',
        f'',
        f'  <link name="{fn}_base">',
        f'    <inertial>',
        f'      <mass value="0.001"/>',
        f'      <origin xyz="0 0 0"/>',
        f'      <inertia ixx="1e-8" iyy="1e-8" izz="1e-8" ixy="0" ixz="0" iyz="0"/>',
        f'    </inertial>',
        f'  </link>',
        f'',
        f'  <!-- MCP curl joint (CONTROLLED by servo) -->',
        f'  <joint name="{fn}_curl" type="revolute">',
        f'    <parent link="{fn}_base"/>',
        f'    <child link="{fn}_knuckle"/>',
        f'    <origin xyz="0 0 0" rpy="0 0 0"/>',
        f'    <axis xyz="{ca[0]:.4f} {ca[1]:.4f} {ca[2]:.4f}"/>',
        f'    <limit lower="{curl_lim[0]}" upper="{curl_lim[1]}" effort="1.0" velocity="3.0"/>',
        f'  </joint>',
        f'',
        f'  <link name="{fn}_knuckle">',
        f'    <visual>',
        f'      <origin xyz="{knuckle_offset[0]:.6f} {knuckle_offset[1]:.6f} {knuckle_offset[2]:.6f}" rpy="0 0 0"/>',
        f'      <geometry><mesh filename="package://hands/urdf/nano/meshes/{knuckle_file}"/></geometry>',
        f'      <material name="{fn}_knuckle_mat"><color rgba="0.85 0.7 0.15 1.0"/></material>',
        f'    </visual>',
        f'    <collision>',
        f'      <origin xyz="{knuckle_offset[0]:.6f} {knuckle_offset[1]:.6f} {knuckle_offset[2]:.6f}" rpy="0 0 0"/>',
        f'      <geometry><mesh filename="package://hands/urdf/nano/meshes/{knuckle_file}"/></geometry>',
        f'    </collision>',
        f'    <inertial>',
        f'      <mass value="0.005"/>',
        f'      <origin xyz="0 0 0"/>',
        f'      <inertia ixx="1e-6" iyy="1e-6" izz="1e-6" ixy="0" ixz="0" iyz="0"/>',
        f'    </inertial>',
        f'  </link>',
        f'',
        f'  <!-- PIP curl joint (MIMIC - follows MCP) -->',
        f'  <!-- UPDATE multiplier={pip_mult} after physical measurement! -->',
        f'  <joint name="{fn}_curl_pip" type="revolute">',
        f'    <parent link="{fn}_knuckle"/>',
        f'    <child link="{fn}_mid"/>',
        f'    <origin xyz="{pip_rel[0]:.6f} {pip_rel[1]:.6f} {pip_rel[2]:.6f}" rpy="0 0 0"/>',
        f'    <axis xyz="{ca[0]:.4f} {ca[1]:.4f} {ca[2]:.4f}"/>',
        f'    <limit lower="{curl_lim[0]}" upper="{curl_lim[1]}" effort="1.0" velocity="3.0"/>',
        f'    <mimic joint="{fn}_curl" multiplier="{pip_mult}" offset="{pip_off}"/>',
        f'  </joint>',
        f'',
        f'  <link name="{fn}_mid">',
        f'    <visual>',
        f'      <origin xyz="{mid_offset[0]:.6f} {mid_offset[1]:.6f} {mid_offset[2]:.6f}" rpy="0 0 0"/>',
        f'      <geometry><mesh filename="package://hands/urdf/nano/meshes/{mid_file}"/></geometry>',
        f'      <material name="{fn}_mid_mat"><color rgba="0.8 0.65 0.12 1.0"/></material>',
        f'    </visual>',
        f'    <collision>',
        f'      <origin xyz="{mid_offset[0]:.6f} {mid_offset[1]:.6f} {mid_offset[2]:.6f}" rpy="0 0 0"/>',
        f'      <geometry><mesh filename="package://hands/urdf/nano/meshes/{mid_file}"/></geometry>',
        f'    </collision>',
        f'    <inertial>',
        f'      <mass value="0.004"/>',
        f'      <origin xyz="0 0 0"/>',
        f'      <inertia ixx="1e-6" iyy="1e-6" izz="1e-6" ixy="0" ixz="0" iyz="0"/>',
        f'    </inertial>',
        f'  </link>',
        f'',
        f'  <!-- DIP curl joint (MIMIC - follows MCP) -->',
        f'  <!-- UPDATE multiplier={dip_mult} after physical measurement! -->',
        f'  <joint name="{fn}_curl_dip" type="revolute">',
        f'    <parent link="{fn}_mid"/>',
        f'    <child link="{fn}_tip"/>',
        f'    <origin xyz="{dip_rel[0]:.6f} {dip_rel[1]:.6f} {dip_rel[2]:.6f}" rpy="0 0 0"/>',
        f'    <axis xyz="{ca[0]:.4f} {ca[1]:.4f} {ca[2]:.4f}"/>',
        f'    <limit lower="{curl_lim[0]}" upper="{curl_lim[1]}" effort="1.0" velocity="3.0"/>',
        f'    <mimic joint="{fn}_curl" multiplier="{dip_mult}" offset="{dip_off}"/>',
        f'  </joint>',
        f'',
        f'  <link name="{fn}_tip">',
        f'    <visual>',
        f'      <origin xyz="{tip_offset[0]:.6f} {tip_offset[1]:.6f} {tip_offset[2]:.6f}" rpy="0 0 0"/>',
        f'      <geometry><mesh filename="package://hands/urdf/nano/meshes/{tip_file}"/></geometry>',
        f'      <material name="{fn}_tip_mat"><color rgba="0.75 0.6 0.1 1.0"/></material>',
        f'    </visual>',
        f'    <collision>',
        f'      <origin xyz="{tip_offset[0]:.6f} {tip_offset[1]:.6f} {tip_offset[2]:.6f}" rpy="0 0 0"/>',
        f'      <geometry><mesh filename="package://hands/urdf/nano/meshes/{tip_file}"/></geometry>',
        f'    </collision>',
        f'    <inertial>',
        f'      <mass value="0.003"/>',
        f'      <origin xyz="0 0 0"/>',
        f'      <inertia ixx="1e-6" iyy="1e-6" izz="1e-6" ixy="0" ixz="0" iyz="0"/>',
        f'    </inertial>',
        f'  </link>',
    ])

lines.extend(['', '</robot>', ''])

urdf_text = '\n'.join(lines)
with open('nanohand.urdf', 'w') as f:
    f.write(urdf_text)

print(f"\n{'='*60}")
print(f"URDF written: nanohand.urdf")
print(f"  Joints per finger: wiggle + curl_mcp + curl_pip(mimic) + curl_dip(mimic)")
print(f"  Total joints: {len([fn for fn in fingers if fn in joint_positions]) * 4}")
print(f"  Controlled DOF: {len([fn for fn in fingers if fn in joint_positions]) * 2} (10)")
print(f"  Mimic joints: {len([fn for fn in fingers if fn in joint_positions]) * 2} (10)")
print(f"")
print(f"  *** IMPORTANT: Mimic multipliers are PLACEHOLDERS! ***")
print(f"  Measure with protractor on real hand using servo GUI.")
print(f"  For each finger, curl to 2-3 angles and measure:")
print(f"    A = MCP angle (proximal)")
print(f"    B = PIP angle → pip_multiplier = B/A")
print(f"    C = DIP angle → dip_multiplier = C/A")
print(f"  Then update MIMIC_VALUES dict in this script and regenerate.")
print(f"")
print(f"Test: ~/urdf-viz nanohand.urdf")
