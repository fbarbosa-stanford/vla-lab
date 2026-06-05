// Shared definition of the tabletop world and the SO-101 arm geometry.
// Units are meters. Arm base sits at the origin on the table (y = 0).
// +X right, +Y up, +Z forward (reaching direction onto the table).

export const CUBE = 0.038;            // cube edge length
const yc = CUBE / 2;                  // cube center height

export const OBJECTS = [
  { id: 'red',    label: 'red cube',    color: 0xe5484d, hex: '#e5484d', pos: [ 0.045, yc,  0.205 ] },
  { id: 'blue',   label: 'blue cube',   color: 0x4d7cff, hex: '#4d7cff', pos: [-0.110, yc,  0.170 ] },
  { id: 'green',  label: 'green cube',  color: 0x3ecf8e, hex: '#3ecf8e', pos: [ 0.150, yc,  0.135 ] },
  { id: 'yellow', label: 'yellow cube', color: 0xf5c542, hex: '#f5c542', pos: [-0.035, yc,  0.110 ] },
];

export function objectById(id) {
  return OBJECTS.find(o => o.id === id) || OBJECTS[0];
}

// SO-101-ish link lengths (meters).
export const ARM = {
  baseH:  0.055,   // table -> shoulder pivot
  L1:     0.118,   // shoulder -> elbow
  L2:     0.108,   // elbow -> wrist
  L3:     0.080,   // wrist -> gripper fingertip
};

// Joint order matches the SO-101 servo chain.
export const JOINTS = [
  'shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper',
];

// A relaxed "home" pose (radians; gripper in normalized 0=closed..1=open).
export const HOME = [0.0, -0.55, 1.15, 0.9, 0.0, 1.0];

// Soft joint limits used for clamping / normalization (radians).
export const LIMITS = [
  [-1.9, 1.9],   // shoulder_pan
  [-1.9, 1.7],   // shoulder_lift
  [-0.2, 2.6],   // elbow_flex
  [-1.8, 1.8],   // wrist_flex
  [-3.1, 3.1],   // wrist_roll
  [ 0.0, 1.0],   // gripper (normalized)
];

export const CHUNK_T = 8;  // action-chunk horizon (timesteps the head predicts at once)
