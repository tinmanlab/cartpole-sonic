import {finiteNumber,integerCount,numericVector,ACTION_FORCE} from "./control_contract.js";
import loadMujoco from "./vendor/mujoco/mujoco.js";

const PRESETS = {
  playground: { label: "MuJoCo Playground", description: "official-style cartpole · dt 0.01 s", cartMass: 1.0, poleMass: 0.1, poleLength: 1.0, sliderRange: 1.8 },
  long: { label: "Long pole", description: "same controller interface · longer inertia", cartMass: 1.0, poleMass: 0.1, poleLength: 1.35, sliderRange: 1.8 },
  heavy: { label: "Heavy pole", description: "same geometry · 2.5× pole mass", cartMass: 1.0, poleMass: 0.25, poleLength: 1.0, sliderRange: 1.8 }
};

function xmlFor(spec) {
  const L = spec.poleLength;
  return [
    '<mujoco model="cart-pole-browser">',
    '  <option timestep="0.01" iterations="1" ls_iterations="4">',
    '    <flag contact="disable" eulerdamp="disable"/>',
    '  </option>',
    '  <default>',
    '    <default class="pole">',
    '      <joint type="hinge" axis="0 1 0" damping="2e-6"/>',
    '      <geom type="capsule" fromto="0 0 0 0 0 '+L+'" size="0.045" mass="'+spec.poleMass+'" rgba=".86 .31 .33 1"/>',
    '    </default>',
    '  </default>',
    '  <worldbody>',
    '    <geom name="floor" pos="0 0 -.05" size="4 4 .2" type="plane" rgba=".95 .96 .98 1"/>',
    '    <geom name="rail1" type="capsule" pos="0 .07 1" zaxis="1 0 0" size="0.02 2" rgba=".75 .78 .82 1"/>',
    '    <geom name="rail2" type="capsule" pos="0 -.07 1" zaxis="1 0 0" size="0.02 2" rgba=".75 .78 .82 1"/>',
    '    <body name="cart" pos="0 0 1">',
    '      <joint name="slider" type="slide" limited="true" axis="1 0 0" range="-'+spec.sliderRange+' '+spec.sliderRange+'" solreflimit=".08 1" damping="5e-4"/>',
    '      <geom name="cart" type="box" size="0.2 0.15 0.1" mass="'+spec.cartMass+'" rgba=".20 .25 .34 1"/>',
    '      <body name="pole_1" childclass="pole">',
    '        <joint name="hinge_1"/>',
    '        <geom name="pole_1"/>',
    '      </body>',
    '    </body>',
    '  </worldbody>',
    '  <actuator>',
    '    <motor name="slide" joint="slider" gear="10" ctrllimited="true" ctrlrange="-1 1"/>',
    '  </actuator>',
    '</mujoco>'
  ].join("\\n");
}

function randomNormal(rng) {
  const u1 = Math.max(1e-12, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export class MuJoCoCartPole {
  constructor() {
    this.mujoco = null;
    this.model = null;
    this.data = null;
    this.preset = "playground";
    this.spec = PRESETS.playground;
    this.backend = "loading";
  }

  static get presets() { return PRESETS; }

  async init(preset = "playground") {
    if (!this.mujoco) this.mujoco = await loadMujoco();
    await this.loadPreset(preset);
    let version = "WASM";
    try { if (typeof this.mujoco.mj_versionString === "function") version = this.mujoco.mj_versionString(); } catch (_) {}
    this.backend = "MuJoCo " + version;
    return this;
  }

  disposeModel() {
    this.data?.delete?.();
    this.model?.delete?.();
    this.data = null;
    this.model = null;
  }

  async loadPreset(preset) {
    if (!PRESETS[preset]) throw new Error("unknown MuJoCo preset: " + preset);
    this.disposeModel();
    this.preset = preset;
    this.spec = PRESETS[preset];
    this.model = this.mujoco.MjModel.from_xml_string(xmlFor(this.spec));
    if (!this.model) throw new Error("MuJoCo failed to compile MJCF");
    this.data = new this.mujoco.MjData(this.model);
    if (!this.data) throw new Error("MuJoCo failed to allocate MjData");
    this.reset({x: 0, theta: 0.10, xDot: 0, thetaDot: 0});
    return this.snapshot();
  }

  reset(options={}) {
    if(!options||typeof options!=="object"||Array.isArray(options))throw new Error("reset state must be an object");
    for(const key of ["x","theta","xDot","thetaDot"])if(Object.hasOwn(options,key))finiteNumber(options[key],key);
    const {x=0,theta=.10,xDot=0,thetaDot=0}=options;
    for(const [name,value] of Object.entries({x,theta,xDot,thetaDot}))finiteNumber(value,name);
    this.mujoco.mj_resetData(this.model,this.data);
    const qpos = this.data.qpos;
    const qvel = this.data.qvel;
    qpos[0] = x; qpos[1] = theta;
    qvel[0] = xDot; qvel[1] = thetaDot;
    if (this.data.ctrl.length) this.data.ctrl[0] = 0;
    this.data.time = 0;
    this.mujoco.mj_forward(this.model, this.data);
    return this.snapshot();
  }

  setState(values) {
    numericVector(values,4,"reset state");
    const [x,xDot,theta,thetaDot]=values;
    return this.reset({x, xDot, theta, thetaDot});
  }

  getState(data = this.data) {
    return [data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]];
  }

  stepForce(forceN, steps=1) {
    finiteNumber(forceN,"forceN");
    const count = integerCount(steps,"physics steps",1,1000);
    const ctrl = Math.max(-1, Math.min(1, forceN / 10));
    for (let i=0; i<count; i++) {
      this.data.ctrl[0] = ctrl;
      this.mujoco.mj_step(this.model, this.data);
    }
    return this.snapshot(forceN);
  }

  applyImpulse(options={}) {
    if(!options||typeof options!=="object"||Array.isArray(options))throw new Error("impulse must be an object");
    for(const key of ["xDotDelta","thetaDotDelta"])if(Object.hasOwn(options,key))finiteNumber(options[key],key);
    const {xDotDelta=.75,thetaDotDelta=-1.25}=options;
    finiteNumber(xDotDelta,"xDotDelta");finiteNumber(thetaDotDelta,"thetaDotDelta");
    this.data.qvel[0] += xDotDelta;
    this.data.qvel[1] += thetaDotDelta;
    this.mujoco.mj_forward(this.model, this.data);
    return this.snapshot();
  }

  snapshot(forceN = null) {
    const [x,xDot,theta,thetaDot] = this.getState();
    return {
      x, xDot, theta, thetaDot,
      time: this.data?.time ?? 0, forceN, requestedForceN:forceN, appliedForceN:forceN===null?null:Math.max(-ACTION_FORCE,Math.min(ACTION_FORCE,forceN)),
      preset: this.preset, model: this.spec.label,
      timestep: this.model?.opt?.timestep ?? 0.01,
      sliderRange: this.spec.sliderRange,
      poleLength: this.spec.poleLength,
      cartMass: this.spec.cartMass,
      poleMass: this.spec.poleMass
    };
  }

  makeData() { return new this.mujoco.MjData(this.model); }

  seedData(data, rng, spread=1) {
    finiteNumber(spread,"spread",0);
    const x=(rng()*2-1)*.10*spread,theta=(rng()*2-1)*.034*spread;
    const xDot=.01*randomNormal(rng)*spread,thetaDot=.01*randomNormal(rng)*spread;
    numericVector([x,xDot,theta,thetaDot],4,"seed state");
    this.mujoco.mj_resetData(this.model,data);
    data.qpos[0]=x;data.qpos[1]=theta;data.qvel[0]=xDot;data.qvel[1]=thetaDot;
    data.ctrl[0] = 0; data.time = 0;
    this.mujoco.mj_forward(this.model, data);
  }

  createDataset(count, scales, rng) {
    const out = new Float32Array(count * 4);
    const data = this.makeData();
    let i = 0;
    this.seedData(data, rng, 3.5);
    try {
      while (i < count) {
        if (Math.abs(data.qpos[0]) > this.spec.sliderRange * 0.94 || Math.abs(data.qpos[1]) > 0.55 ||
            !Number.isFinite(data.qpos[0] + data.qpos[1] + data.qvel[0] + data.qvel[1])) {
          this.seedData(data, rng, 3.5);
        }
        const s = this.getState(data);
        for (let j=0;j<4;j++) out[i*4+j] = Math.max(-1, Math.min(1, s[j] / scales[j]));
        i++;
        const exploratoryForce = (rng()*2-1) * 7.5;
        data.ctrl[0] = exploratoryForce / 10;
        this.mujoco.mj_step(this.model, data);
      }
    } finally {
      data.delete();
    }
    return out;
  }

  dispose() { this.disposeModel(); }
}
