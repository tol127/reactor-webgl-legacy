'use strict';

/* ==================== DOM ==================== */
const $ = id => document.getElementById(id);
const canvas = $('core');
const spec = $('spectrum');
const sg = spec.getContext('2d');

/* ==================== 状态 ==================== */
let ctx, analyser, bins, wave, master, stream, source, buffer;
let started = 0, offset = 0, mode = 'demo', playing = false, loading = false;
let session = 0, rafTime = 0;
let angle = -0.32, tilt = 0.58, zoom = 1, drag = null;
let demoNodes = [];
let gain = 1.4, smoothing = 0.75, energy = 0;
let smoothPeakIndex = 0;
let seeking = false;

/* ==================== 控制棒数据 ==================== */
const rods = [];
for (let x = -15; x <= 15; x++)
  for (let z = -15; z <= 15; z++) {
    const r = Math.hypot(x, z);
    if (r <= 15) rods.push({
      x, z, r,
      h: 0,
      seed: Math.abs(Math.sin(x * 127.1 + z * 311.7))
    });
  }
$('count').textContent = rods.length;
const N = rods.length;

/* ==================== WebGL 初始化 ==================== */
const gl = canvas.getContext('webgl', {
  antialias: true,
  alpha: true,
  premultipliedAlpha: true,
  powerPreference: 'high-performance'
});
if (!gl) {
  document.body.innerHTML = '<p style="color:#fff;padding:40px;font-family:sans-serif">当前浏览器不支持 WebGL。</p>';
  throw new Error('no webgl');
}

const ext = gl.getExtension('ANGLE_instanced_arrays');
if (!ext) {
  document.body.innerHTML = '<p style="color:#fff;padding:40px;font-family:sans-serif">当前浏览器不支持 WebGL instancing。</p>';
  throw new Error('no instancing');
}

/* ---------- Shader ---------- */
/* 正交投影 + 等距变换在 CPU 端完成，GPU 只负责把顶点变换到 clip 空间。
   每个顶点自带世界坐标和颜色，CPU 每帧更新。 */
const VS = `
attribute vec3 aPos;      // 已投影到裁剪空间前的屏幕坐标（-1..1 区间由 CPU 算好）
attribute vec3 aColor;    // 每顶点颜色
varying vec3 vColor;
void main(){
  vColor = aColor;
  gl_Position = vec4(aPos, 1.0);
}`;

const FS = `
precision mediump float;
varying vec3 vColor;
void main(){
  gl_FragColor = vec4(vColor, 1.0);
}`;

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error('Shader compile error: ' + gl.getShaderInfoLog(s));
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
  throw new Error('Program link error: ' + gl.getProgramInfoLog(prog));
gl.useProgram(prog);

const A = {
  pos:   gl.getAttribLocation(prog, 'aPos'),
  color: gl.getAttribLocation(prog, 'aColor')
};

/* ==================== 顶点缓冲 ==================== */
/* 每一帧 CPU 侧把所有几何顶点算好，写进一个大 Float32Array，然后一次 drawArrays。
   顶点数估算：
   - 柱子：709 × 5 面 × 2 三角 × 3 顶点 = 21270 顶点（侧面 + 顶面）
   - 柱子中心点：709 × 1 面 × 6 顶点（用两个三角形拼一个菱形近似圆点）= 4254
   - 环：4 层 × 96 段 × 2 三角 × 3 = 2304
   - 刻度块：80 × 2 三角 × 3 = 480
   - 外圈细线环：几个环，各 96 段 × 2 × 3 ≈ 几百
   总计约 29000 顶点，每顶点 6 float（xyz + rgb）= 174000 float = 696KB，每帧上传一次，没问题。
*/
const ROD_VERTS = 30;         // 4 侧面(24) + 1 顶面(6)
const ROD_DOT_VERTS = 6;      // 中心点（菱形两三角）
const RING_SEGMENTS = 96;
const RING_VERTS_PER_RING = RING_SEGMENTS * 6;
const RING_COUNT = 4;
const TICK_COUNT = 80;
const TICK_VERTS = TICK_COUNT * 6;
const OUTLINE_RING_COUNT = 4; // 额外几个细线环，模仿原版 ring() 调用

//const MAX_VERTS =
//  N * (ROD_VERTS + ROD_DOT_VERTS) +
//  RING_COUNT * RING_VERTS_PER_RING +
//  TICK_VERTS +
//  OUTLINE_RING_COUNT * RING_VERTS_PER_RING;  //不够，最少也得60000 VERTS
const MAX_VERTS =80000

const vertexData = new Float32Array(MAX_VERTS * 6);   // x,y,z,r,g,b
let vertexCount = 0;

const vbo = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
gl.bufferData(gl.ARRAY_BUFFER, vertexData.byteLength, gl.DYNAMIC_DRAW);

/* ==================== 等距投影（和原版 point() 完全一致） ==================== */
let canvasW = 1, canvasH = 1;

function project(x, z, h) {
  // 原版：
  // const s = Math.min(w/45, H/37) * zoom;
  // rx = x*cos(angle) - z*sin(angle)
  // rz = x*sin(angle) + z*cos(angle)
  // return [w/2 + rx*s, H*0.55 + rz*s*tilt - h*s]
  const s = Math.min(canvasW / 45, canvasH / 37) * zoom;
  const ca = Math.cos(angle), sa = Math.sin(angle);
  const rx = x * ca - z * sa;
  const rz = x * sa + z * ca;
  // 返回像素坐标
  return [
    canvasW / 2 + rx * s,
    canvasH * 0.55 + rz * s * tilt - h * s
  ];
}

/* 把像素坐标转为 WebGL clip 空间（-1..1，y 向上） */
function toClip(px, py) {
  return [
    (px / canvasW) * 2 - 1,
    1 - (py / canvasH) * 2
  ];
}

/* ==================== 颜色工具 ==================== */
function hex2rgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
function rgb(r, g, b) {
  return [r / 255, g / 255, b / 255];
}

const COL = {
  ringOuterDark:  hex2rgb('#060b07'),
  ringMain:       hex2rgb('#505948'),
  ringInner:      hex2rgb('#878969'),
  ringOuter:      hex2rgb('#303a2d'),
  tickA:          hex2rgb('#484d31'),
  tickB:          hex2rgb('#adac64'),
  side1:          hex2rgb('#424838'),
  side3:          hex2rgb('#333c2b'),
  side4:          hex2rgb('#39432f'),
  sideEdge:       hex2rgb('#111910'),
  topEdge:        hex2rgb('#232d20'),
  rodDot:         hex2rgb('#404735'),
  rodDotPin:      [1, 1, 1]
};

/* ==================== 几何生成：每帧重算顶点 ==================== */
let vp = 0;   // vertexData 写指针（float 索引）

function pushVertex(px, py, col) {
  const c = toClip(px, py);
  vertexData[vp++] = c[0];
  vertexData[vp++] = c[1];
  vertexData[vp++] = 0;
  vertexData[vp++] = col[0];
  vertexData[vp++] = col[1];
  vertexData[vp++] = col[2];
}

function pushEdge(p1, p2, col, EW) {
  const dx = p2[0] - p1[0], dy = p2[1] - p1[1];
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len * EW * 0.5;
  const ny =  dx / len * EW * 0.5;
  const A = [p1[0] + nx, p1[1] + ny];
  const B = [p2[0] + nx, p2[1] + ny];
  const C = [p2[0] - nx, p2[1] - ny];
  const D = [p1[0] - nx, p1[1] - ny];
  pushVertex(A[0], A[1], col);
  pushVertex(B[0], B[1], col);
  pushVertex(C[0], C[1], col);
  pushVertex(A[0], A[1], col);
  pushVertex(C[0], C[1], col);
  pushVertex(D[0], D[1], col);
}

function pushQuad(p1, p2, p3, p4, col) {
  // p1..p4 是像素坐标数组 [x,y]，顺序为逆时针
  pushVertex(p1[0], p1[1], col);
  pushVertex(p2[0], p2[1], col);
  pushVertex(p3[0], p3[1], col);
  pushVertex(p1[0], p1[1], col);
  pushVertex(p3[0], p3[1], col);
  pushVertex(p4[0], p4[1], col);
  
}

/* ==================== 尺寸 ==================== */
function size() {
  const rect = canvas.getBoundingClientRect();
  const d = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * d));
  canvas.height = Math.max(1, Math.round(rect.height * d));
  canvasW = canvas.clientWidth;
  canvasH = canvas.clientHeight;
  gl.viewport(0, 0, canvas.width, canvas.height);

  const srect = spec.getBoundingClientRect();
  spec.width = Math.max(1, Math.round(srect.width * d));
  spec.height = Math.max(1, Math.round(srect.height * d));
  sg.setTransform(d, 0, 0, d, 0, 0);
  
}
new ResizeObserver(size).observe(canvas);
new ResizeObserver(size).observe(spec);

/* ==================== 主循环 ==================== */
function frame(t) {
  const dt = Math.min((t - rafTime) / 16.67, 3) || 1;
  rafTime = t;

  // ---- 音频分析 ----
  if (analyser && playing) analyser.getByteFrequencyData(bins);
  //let peak = 0, peakIndex = 0;
  //if (playing && bins) {
  //  for (let i = 4; i < bins.length; i++) {
  //    if (bins[i] > peak) { peak = bins[i]; peakIndex = i; }
  //  }
  //}
  let peak = 0, peakIndex = 0;
  if (playing && bins) {
  for (let i = 2; i < bins.length - 2; i++) {
    if (bins[i] > peak) { peak = bins[i]; peakIndex = i; }
  }
  smoothPeakIndex = smoothPeakIndex * 0.7 + peakIndex * 0.3;
  }
  energy = 0;
  if (playing && wave) {
    analyser.getFloatTimeDomainData(wave);
    for (let i = 0; i < wave.length; i++) energy += wave[i] * wave[i];
    energy = Math.sqrt(energy / wave.length);
  }

  // ---- 更新每根柱子高度 ----
  const sampleRate = ctx ? ctx.sampleRate : 44100;
  const binHz = sampleRate / 2048;

  for (let i = 0; i < N; i++) {
    const rod = rods[i];
    let amp = 0;
    if (playing && bins) {
      const hz = 45 * Math.pow(320, rod.r / 15);
      const idx = Math.max(1, Math.min(bins.length - 2, Math.round(hz / binHz)));
      amp = (bins[idx] + bins[idx - 1] + bins[idx + 1]) / 765;
      amp = Math.pow(amp, 1.7) * gain;
    } else if (!playing && mode === 'demo' && !ctx) {
      amp = 0.10 + 0.20 * (0.5 + 0.5 * Math.sin(rod.r * 0.6 - t * 0.0007 + rod.x * 0.1));
    }
    rod._amp = amp;
    const target = 0.07 + amp * 4.8;
    rod.h += (target - rod.h) * (1 - Math.pow(0.45 + smoothing * 0.53, dt));
  }

  // ---- 写顶点 ----
  vp = 0;

  // ===== 1. 外圈环（先画，作为背景） =====
  // 原版 ring() 用 100 段画线，这里用很细的四边形近似
  function drawRing(radius, colorHex, widthPx) {
    const col = hex2rgb(colorHex);
    const s = Math.min(canvasW / 45, canvasH / 37) * zoom;
    const halfW = (widthPx * 0.5) / s;   // 半径差 = 像素宽 / 缩放
    const step = RING_SEGMENTS;
    for (let i = 0; i < step; i++) {
      const a0 = i / step * Math.PI * 2;
      const a1 = (i + 1) / step * Math.PI * 2;
      const c0 = Math.cos(a0), s0 = Math.sin(a0);
      const c1 = Math.cos(a1), s1 = Math.sin(a1);
      const rIn = radius - halfW;
      const rOut = radius + halfW;
      const p0i = project(c0 * rIn, s0 * rIn, 0);
      const p1i = project(c1 * rIn, s1 * rIn, 0);
      const p0o = project(c0 * rOut, s0 * rOut, 0);
      const p1o = project(c1 * rOut, s1 * rOut, 0);
      pushQuad(p0i, p1i, p1o, p0o, col);
    }
  }

  // 原版顺序：ring(16.4,'#060b07',20); ring(16.5,'#505948',2); ring(16,'#878969',1); ring(17.3,'#303a2d',1);
  drawRing(16.4, '#060b07', 60 * (Math.min(canvasW / 45, canvasH / 37) * zoom) * 0.02);
  drawRing(16.5, '#505948', 6 * (Math.min(canvasW / 45, canvasH / 37) * zoom) * 0.02);
  drawRing(16.0, '#878969', 3 * (Math.min(canvasW / 45, canvasH / 37) * zoom) * 0.02);
  drawRing(17.3, '#303a2d', 3 * (Math.min(canvasW / 45, canvasH / 37) * zoom) * 0.02);

  // ===== 2. 80 个刻度块 =====
  for (let i = 0; i < 80; i++) {
    const a = i / 80 * Math.PI * 2;
    const a2 = a + 0.034;
    const r0 = 15.7, r1 = 16.15;
    const p1 = project(Math.cos(a) * r0, Math.sin(a) * r0, 0);
    const p2 = project(Math.cos(a2) * r0, Math.sin(a2) * r0, 0);
    const p3 = project(Math.cos(a2) * r1, Math.sin(a2) * r1, 0);
    const p4 = project(Math.cos(a) * r1, Math.sin(a) * r1, 0);
    const col = (i % 2 === 1) ? COL.tickA : COL.tickB;
    pushQuad(p1, p2, p3, p4, col);
    
  }

  // ===== 3. 柱子（按深度排序，远的先画） =====
  const sorted = [...rods].sort((a, b) =>
    (a.x * Math.sin(angle) + a.z * Math.cos(angle)) -
    (b.x * Math.sin(angle) + b.z * Math.cos(angle))
  );

  const S = 0.44;
  const dotSize = Math.min(canvasW / 45, canvasH / 37) * zoom * 0.12;

  for (const rod of sorted) {
    const amp = rod._amp || 0;
    const h = rod.h;
    const x = rod.x, z = rod.z;

    // 8 个顶点（底面 4 + 顶面 4）
    const a0 = project(x - S, z - S, 0);
    const b0 = project(x + S, z - S, 0);
    const c0 = project(x + S, z + S, 0);
    const d0 = project(x - S, z + S, 0);
    const a  = project(x - S, z - S, h);
    const b  = project(x + S, z - S, h);
    const c  = project(x + S, z + S, h);
    const d  = project(x - S, z + S, h);

    // 侧面 1 (a-a0-b0-b)：固定 #424838
    pushQuad(a, b, b0, a0, COL.side1);
    // 侧面 2 (b-b0-c0-c)：动态
    pushQuad(b, c, c0, b0, rgb(
      55 + amp * 30,
      62 + amp * 30,
      44 + amp * 20
    ));
    // 侧面 3 (c-c0-d0-d)：固定 #333c2b
    pushQuad(c, d, d0, c0, COL.side3);
    // 侧面 4 (d-d0-a0-a)：固定 #39432f
    pushQuad(d, a, a0, d0, COL.side4);

    // 顶面颜色
    const s = rod.seed;
    let base;
    if (s > 0.92) base = [143, 109, 81];
    else if (s > 0.85) base = [97, 118, 104];
    else base = [139 + s * 29, 148 + s * 22, 119 + s * 21];
    pushQuad(a, b, c, d, rgb(
      base[0] + amp * 43,
      base[1] + amp * 40,
      base[2] + amp * 24
    ));
      // 边框
    const EW = 0.8;
    const EC = hex2rgb('#0d1409');
    // 顶面 4 边
    pushEdge(a, b, EC, EW);
    pushEdge(b, c, EC, EW);
    pushEdge(c, d, EC, EW);
    pushEdge(d, a, EC, EW);
    // 竖棱 4 条 (多画后面两条，不要)
    // pushEdge(a, a0, EC, EW);
    //pushEdge(b, b0, EC, EW);
    //pushEdge(c, c0, EC, EW);
    //pushEdge(d, d0, EC, EW);
  
    // ---- 竖棱（只画朝镜头的两根） ----
    const sa = Math.sin(angle), ca = Math.cos(angle);
    const rzA = -sa - ca;   // a 角点
    const rzB =  sa - ca;   // b 角点
    const rzC =  sa + ca;   // c 角点
    const rzD = -sa + ca;   // d 角点

    // 找最大的两个
    const arr = [
      { rz: rzA, p1: a, p2: a0 },
      { rz: rzB, p1: b, p2: b0 },
      { rz: rzC, p1: c, p2: c0 },
      { rz: rzD, p1: d, p2: d0 }
    ];
    arr.sort((m, n) => n.rz - m.rz);
    pushEdge(arr[0].p1, arr[0].p2, EC, EW);
    pushEdge(arr[1].p1, arr[1].p2, EC, EW);

    // 柱子中心点（近似原版椭圆）
    const cp = project(x, z, h + 0.01);
    const px = cp[0], py = cp[1];
    const col = COL.rodDot;
    const rx = dotSize, ry = dotSize * tilt;
    // 用两个三角形拼菱形
    pushVertex(px - rx, py, col);
    pushVertex(px, py - ry, col);
    pushVertex(px + rx, py, col);
    pushVertex(px - rx, py, col);
    pushVertex(px + rx, py, col);
    pushVertex(px, py + ry, col);

    // 小亮点
    //if (dotSize > 13 && s > 0.6) {
    //  const pin = COL.rodDotPin;
    //  pushVertex(px - 0.5, py - 0.5, pin);
    //  pushVertex(px + 0.5, py - 0.5, pin);
    //  pushVertex(px + 0.5, py + 0.5, pin);
    //  pushVertex(px - 0.5, py - 0.5, pin);
    //  pushVertex(px + 0.5, py + 0.5, pin);
    //  pushVertex(px - 0.5, py + 0.5, pin);
    //}
  }

  vertexCount = vp / 6;

  // ---- 上传 ----
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, vertexData.subarray(0, vp));

  // ---- 绘制 ----
  //gl.clearColor(0.071, 0.086, 0.078, 1);
  //gl.clearColor(0, 0, 0, 1);
  gl.disable(gl.DEPTH_TEST);   // 已按深度排序，不需要 z-buffer
  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.enableVertexAttribArray(A.pos);
  gl.vertexAttribPointer(A.pos, 3, gl.FLOAT, false, 24, 0);
  gl.enableVertexAttribArray(A.color);
  gl.vertexAttribPointer(A.color, 3, gl.FLOAT, false, 24, 12);

  gl.drawArrays(gl.TRIANGLES, 0, vertexCount);

  // ---- 频谱小图 ----
  const sw = spec.clientWidth, sh = spec.clientHeight;
  sg.clearRect(0, 0, sw, sh);
  for (let i = 0; i < 48; i++) {
    const v = playing && bins
      ? bins[Math.min(bins.length - 1, Math.floor(Math.pow(i / 47, 2) * 700))] / 255
      : 0;
    sg.fillStyle = i < 12 ? '#d8b975' : i < 30 ? '#c3cf87' : '#95b1a1';
    sg.fillRect(i * sw / 48, sh - Math.max(2, v * sh), sw / 48 - 3, Math.max(2, v * sh));
  }

  // ---- 读数 ----
  if (t % 100 < 25) {
    $('level').innerHTML = (playing ? Math.round(20 * Math.log10(Math.max(energy, 0.001))) : '—') + '<small>dB</small>';
    $('freq').innerHTML = (playing && peak > 0? Math.round(smoothPeakIndex * binHz): '—') + '<small>Hz</small>';
    if (playing && !seeking) {
      const seconds = offset + ctx.currentTime - started;
      $('elapsed').textContent = format(seconds);
      if (buffer && mode === 'file') $('seek').value = seconds / buffer.duration * 100;
    }
  }

  requestAnimationFrame(frame);
}

/* ==================== 音频系统 ==================== */
function format(s) {
  return Math.floor(s / 60).toString().padStart(2, '0') + ':' +
         Math.floor(s % 60).toString().padStart(2, '0');
}

async function audio() {
  if (!ctx) {
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.75;
    bins = new Uint8Array(analyser.frequencyBinCount);
    wave = new Float32Array(analyser.fftSize);
    master = ctx.createGain();
    master.gain.value = 0.4;
    analyser.connect(master);
    master.connect(ctx.destination);
  }
  await ctx.resume();
}

function stop() {
  session++;
  loading = false;
  if (source) {
    source.onended = null;
    try { source.stop?.(); } catch (e) {}
    source.disconnect();
    source = null;
  }
  demoNodes.forEach(n => { try { n.stop?.(); n.disconnect(); } catch (e) {} });
  demoNodes = [];
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  playing = false;
  update();
}

function update() {
  $('play').textContent = playing ? 'Ⅱ' : '▶';
  $('play').setAttribute('aria-label', playing ? '暂停' : '播放');
  $('demo').classList.toggle('selected', mode === 'demo');
  $('mic').classList.toggle('selected', mode === 'mic');
  $('status').textContent = playing
    ? '运行中 · ' + (mode === 'demo' ? '演示信号' : mode === 'mic' ? '实时输入' : '音频分析')
    : '待机 · ' + (mode === 'demo' ? '演示预览' : '已暂停');
  $('tracksub').textContent = playing ? '控制棒随频谱实时响应' : '点击播放，启动声场';
  $('seek').disabled = mode !== 'file';
}

async function start() {
  if (loading || playing) return;
  loading = true;
  const attempt = ++session;
  try {
    await audio();
    if (attempt !== session) return;

    if (mode === 'mic') {
      const id = ++session;
      loading = true;
      $('message').textContent = '等待麦克风授权…';
      const s = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (id !== session) {
        s.getTracks().forEach(t => t.stop());
        return;
      }
      stream = s;
      source = ctx.createMediaStreamSource(s);
      source.connect(analyser);
      master.gain.value = 0;
      $('message').textContent = '正在监听麦克风；停止或切换输入即可关闭。';
    } else {
      master.gain.value = +$('volume').value / 100;

      if (mode === 'file' && buffer) {
        if (offset >= buffer.duration) offset = 0;
        source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(analyser);
        source.onended = () => {
          stop();
          offset = 0;
          $('seek').value = 0;
          $('elapsed').textContent = '00:00';
        };
        source.start(0, offset);
      } else {
        [55, 82.41, 110, 164.81, 220, 329.63, 440, 659.25].forEach((freq, i) => {
          const osc = ctx.createOscillator();
          const v = ctx.createGain();
          const lfo = ctx.createOscillator();
          const lg = ctx.createGain();
          osc.type = i < 3 ? 'triangle' : 'sine';
          osc.frequency.value = freq;
          v.gain.value = 0.025;
          lfo.frequency.value = 0.22 + i * 0.087;
          lg.gain.value = 0.023;
          lfo.connect(lg);
          lg.connect(v.gain);
          osc.connect(v);
          v.connect(analyser);
          osc.start();
          lfo.start();
          demoNodes.push(osc, lfo, v, lg);
        });
      }
      $('message').textContent = '音频仅在本机处理，不会上传。';
    }

    started = ctx.currentTime;
    playing = true;
    update();
  } catch (e) {
    stop();
    $('message').textContent = mode === 'mic'
      ? '无法访问麦克风，请检查浏览器权限，或导入音频。'
      : '无法播放此音频，请尝试其他文件。';
  } finally {
    if (attempt === session || !playing) loading = false;
  }
}

/* ==================== 事件绑定 ==================== */
$('play').onclick = () => {
  if (playing) {
    offset += ctx.currentTime - started;
    stop();
  } else start();
};

$('demo').onclick = () => {
  stop();
  mode = 'demo';
  offset = 0;
  buffer = null;
  $('trackname').textContent = 'REACTOR / 内置信号';
  $('duration').textContent = 'DEMO / LOOP';
  $('elapsed').textContent = '00:00';
  $('seek').value = 0;
  update();
  start();
};

$('mic').onclick = () => {
  stop();
  mode = 'mic';
  offset = 0;
  buffer = null;
  $('trackname').textContent = 'MIC / 麦克风输入';
  $('duration').textContent = 'LIVE';
  $('elapsed').textContent = '00:00';
  update();
  start();
};

$('upload').onclick = () => $('file').click();

$('file').onchange = async e => {
  const f = e.target.files[0];
  if (!f) return;
  stop();
  const id = session;
  loading = true;
  $('message').textContent = '正在读取音频…';
  try {
    await audio();
    const decoded = await ctx.decodeAudioData(await f.arrayBuffer());
    if (id !== session) return;
    buffer = decoded;
    mode = 'file';
    offset = 0;
    $('trackname').textContent = f.name;
    $('duration').textContent = format(buffer.duration);
    $('elapsed').textContent = '00:00';
    $('seek').value = 0;
    update();
    loading = false;
    await start();
  } catch (err) {
    $('message').textContent = '无法读取该文件，请尝试 MP3 或 WAV 音频。';
  } finally {
    loading = false;
    e.target.value = '';
  }
};

$('seek').onkeydown = e => {
  if (mode !== 'file' || !buffer) return;
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  e.preventDefault();

  const STEP = 5;                                   // 每次 5 秒
  const dir = e.key === 'ArrowRight' ? 1 : -1;
  const cur = +$('seek').value / 100 * buffer.duration;
  const next = Math.max(0, Math.min(buffer.duration, cur + dir * STEP));

  $('seek').value = next / buffer.duration * 100;
  $('seek').dispatchEvent(new Event('input'));      // 更新 elapsed 显示
  $('seek').dispatchEvent(new Event('change'));     // 触发真正的 seek
};

$('seek').oninput = () => {
  if (mode !== 'file' || !buffer) return;
  seeking = true;                                    // 拖动开始
  $('elapsed').textContent = format(+$('seek').value / 100 * buffer.duration);
};

$('seek').onchange = () => {
  if (mode !== 'file' || !buffer) return;
  const was = playing;
  stop();
  offset = +$('seek').value / 100 * buffer.duration;
  $('elapsed').textContent = format(offset);
  seeking = false;
  if (was) start();
};

$('gain').oninput = e => {
  gain = +e.target.value;
  $('gainval').textContent = gain.toFixed(1) + '×';
};

$('smooth').oninput = e => {
  smoothing = +e.target.value / 100;
  $('smoothval').textContent = e.target.value + '%';
};

$('volume').oninput = e => {
  $('volumeval').textContent = e.target.value + '%';
  if (master && mode !== 'mic')
    master.gain.setTargetAtTime(+e.target.value / 100, ctx.currentTime, 0.03);
};

/* 拖拽旋转 */
canvas.addEventListener('pointerdown', e => {
  drag = { x: e.clientX, y: e.clientY };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', e => {
  if (!drag) return;
  angle -= (e.clientX - drag.x) * 0.006;
  tilt = Math.max(0.3, Math.min(0.8, tilt + (e.clientY - drag.y) * 0.005));
  drag = { x: e.clientX, y: e.clientY };
});
canvas.addEventListener('pointerup', () => drag = null);
canvas.addEventListener('pointercancel', () => drag = null);

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  zoom = Math.max(0.5, Math.min(2.2, zoom - e.deltaY * 0.001));
}, { passive: false });

$('reset').onclick = () => {
  angle = -0.32;
  tilt = 0.58;
  zoom = 1;
};

$('fullscreen').onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch (e) {
    $('message').textContent = '当前浏览器不支持全屏。';
  }
};

document.addEventListener('fullscreenchange', () => {
  $('fullscreen').textContent = document.fullscreenElement ? '退出全屏 ↙' : '全屏 ↗';
});

window.addEventListener('pagehide', stop);

/* ==================== 启动 ==================== */
size();
requestAnimationFrame(frame);
update();

/* ==================== 可选：Model Context 工具注册 ====================  */
if (document.modelContext?.registerTool) {
  try {
    Promise.resolve(document.modelContext.registerTool({
      name: 'configure_reactor_response',
      description: 'Adjust the reactor audio visualizer sensitivity and motion smoothing.',
      inputSchema: {
        type: 'object',
        properties: {
          sensitivity: { type: 'number', minimum: 0.3, maximum: 3 },
          smoothing: { type: 'number', minimum: 0, maximum: 95 }
        },
        required: ['sensitivity', 'smoothing'],
        additionalProperties: false
      },
      execute(input) {
        if (!input ||
            !Number.isFinite(input.sensitivity) || input.sensitivity < 0.3 || input.sensitivity > 3 ||
            !Number.isFinite(input.smoothing) || input.smoothing < 0 || input.smoothing > 95)
          throw Error('Invalid response settings');
        $('gain').value = input.sensitivity;
        $('gain').dispatchEvent(new Event('input'));
        $('smooth').value = input.smoothing;
        $('smooth').dispatchEvent(new Event('input'));
        return { sensitivity: gain, smoothing: smoothing * 100 };
      }
    })).catch(() => {});
  } catch (e) {}
}