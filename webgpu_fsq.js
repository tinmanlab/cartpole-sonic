export async function probeWebGPUFSQ() {
  if (!navigator.gpu) return {available:false, ok:false, reason:"navigator.gpu unavailable"};
  const adapter = await Promise.race([
    navigator.gpu.requestAdapter({powerPreference:"high-performance"}),
    new Promise((_,reject)=>setTimeout(()=>reject(new Error("adapter timeout")),1800))
  ]).catch(()=>null);
  if (!adapter) return {available:false, ok:false, reason:"adapter unavailable"};

  const device = await Promise.race([
    adapter.requestDevice(),
    new Promise((_,reject)=>setTimeout(()=>reject(new Error("device timeout")),1800))
  ]).catch(()=>null);
  if (!device) return {available:true, ok:false, reason:"device unavailable"};

  const input = new Float32Array([-1.2,-0.7,-0.2,0,0.18,0.49,0.9,1.4]);
  const bytes = input.byteLength;
  const inBuf = device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const outBuf = device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const readBuf = device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  device.queue.writeBuffer(inBuf,0,input);

  const shader = device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> src: array<f32>;
@group(0) @binding(1) var<storage,read_write> dst: array<f32>;
@compute @workgroup_size(8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= 8u) { return; }
  dst[i] = round(tanh(src[i]) * 1.998) / 2.0;
}`});
  const pipeline = device.createComputePipeline({layout:"auto",compute:{module:shader,entryPoint:"main"}});
  const bind = device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
    {binding:0,resource:{buffer:inBuf}},
    {binding:1,resource:{buffer:outBuf}}
  ]});
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0,bind); pass.dispatchWorkgroups(1); pass.end();
  encoder.copyBufferToBuffer(outBuf,0,readBuf,0,bytes);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readBuf.mapAsync(GPUMapMode.READ);
  const gpu = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();

  const cpu = Array.from(input, v => Math.round(Math.tanh(v)*1.998)/2);
  const maxError = Math.max(...cpu.map((v,i)=>Math.abs(v-gpu[i])));
  const info = adapter.info || {};
  const label = [info.vendor,info.architecture,info.device].filter(Boolean).join(" · ") || "browser WebGPU adapter";

  inBuf.destroy(); outBuf.destroy(); readBuf.destroy(); device.destroy?.();
  return {available:true,ok:maxError < 1e-6,label,maxError,input:Array.from(input),output:Array.from(gpu)};
}
