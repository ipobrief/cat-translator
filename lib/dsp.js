// 브라우저 + Node 공용 오디오 특징 추출 (학습/추론 동일 코드로 특징 일치 보장)
// 출력: 고정 길이 특징 벡터 (Float64Array)
export const TARGET_SR = 16000;
const WIN = 400, HOP = 160, NFFT = 512, N_MEL = 26, N_MFCC = 13;
const FMIN = 0, FMAX = 8000;

// ---- 선형 리샘플 (브라우저/Node 동일) ----
export function resample(x, srcSR, dstSR = TARGET_SR) {
  if (srcSR === dstSR) return Float64Array.from(x);
  const ratio = dstSR / srcSR;
  const n = Math.round(x.length * ratio);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / ratio, i0 = Math.floor(t), frac = t - i0;
    const a = x[i0] || 0, b = x[i0 + 1] !== undefined ? x[i0 + 1] : a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

// ---- 무음 트림 (간단 에너지 기반) ----
function trim(x, thresh = 0.005) {
  let s = 0, e = x.length - 1;
  while (s < x.length && Math.abs(x[s]) < thresh) s++;
  while (e > s && Math.abs(x[e]) < thresh) e--;
  if (e <= s) return x;
  return x.slice(s, e + 1);
}

// ---- 반복 radix-2 FFT (실수 입력) ----
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = i + k + len / 2;
        const tr = cr * re[b] - ci * im[b], ti = cr * im[b] + ci * re[b];
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

const hann = new Float64Array(WIN);
for (let i = 0; i < WIN; i++) hann[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (WIN - 1));

function hz2mel(f) { return 2595 * Math.log10(1 + f / 700); }
function mel2hz(m) { return 700 * (10 ** (m / 2595) - 1); }

// 멜 필터뱅크
const melFb = (() => {
  const bins = NFFT / 2 + 1;
  const mMin = hz2mel(FMIN), mMax = hz2mel(FMAX);
  const pts = new Array(N_MEL + 2);
  for (let i = 0; i < pts.length; i++)
    pts[i] = Math.floor((NFFT + 1) * mel2hz(mMin + (mMax - mMin) * i / (N_MEL + 1)) / TARGET_SR);
  const fb = Array.from({ length: N_MEL }, () => new Float64Array(bins));
  for (let m = 1; m <= N_MEL; m++) {
    for (let k = pts[m - 1]; k < pts[m]; k++)
      if (pts[m] > pts[m - 1]) fb[m - 1][k] = (k - pts[m - 1]) / (pts[m] - pts[m - 1]);
    for (let k = pts[m]; k < pts[m + 1]; k++)
      if (pts[m + 1] > pts[m]) fb[m - 1][k] = (pts[m + 1] - k) / (pts[m + 1] - pts[m]);
  }
  return fb;
})();

// DCT-II 행렬
const dctM = Array.from({ length: N_MFCC }, (_, k) =>
  Float64Array.from({ length: N_MEL }, (_, n) =>
    Math.cos(Math.PI * k * (2 * n + 1) / (2 * N_MEL))));

function stats(arr) {
  let m = 0; for (const v of arr) m += v; m /= arr.length;
  let s = 0; for (const v of arr) s += (v - m) ** 2; s = Math.sqrt(s / arr.length);
  return [m, s];
}

// ---- 메인: PCM(Float) + srcSR -> 특징 벡터 ----
export function extractFeatures(pcm, srcSR) {
  let y = resample(pcm, srcSR, TARGET_SR);
  y = trim(y);
  if (y.length < TARGET_SR / 10) {
    const pad = new Float64Array(TARGET_SR / 10); pad.set(y); y = pad;
  }
  const mfccFrames = [], cent = [], band = [], roll = [], zcr = [], rms = [];
  const bins = NFFT / 2 + 1;
  for (let start = 0; start + WIN <= y.length; start += HOP) {
    const re = new Float64Array(NFFT), im = new Float64Array(NFFT);
    let energy = 0, zc = 0;
    for (let i = 0; i < WIN; i++) {
      const s = y[start + i];
      re[i] = s * hann[i];
      energy += s * s;
      if (i > 0 && ((s >= 0) !== (y[start + i - 1] >= 0))) zc++;
    }
    fft(re, im);
    const power = new Float64Array(bins);
    let psum = 0, fcent = 0;
    for (let k = 0; k < bins; k++) {
      power[k] = (re[k] * re[k] + im[k] * im[k]) / NFFT;
      const f = k * TARGET_SR / NFFT;
      psum += power[k]; fcent += f * power[k];
    }
    const centroid = psum > 0 ? fcent / psum : 0;
    let bw = 0; for (let k = 0; k < bins; k++) {
      const f = k * TARGET_SR / NFFT; bw += power[k] * (f - centroid) ** 2;
    }
    bw = psum > 0 ? Math.sqrt(bw / psum) : 0;
    let cum = 0, rolloff = 0; const thr = 0.85 * psum;
    for (let k = 0; k < bins; k++) { cum += power[k]; if (cum >= thr) { rolloff = k * TARGET_SR / NFFT; break; } }

    // 멜 -> log -> DCT
    const mel = new Float64Array(N_MEL);
    for (let m = 0; m < N_MEL; m++) {
      let acc = 0; const fb = melFb[m];
      for (let k = 0; k < bins; k++) acc += fb[k] * power[k];
      mel[m] = Math.log(acc + 1e-10);
    }
    const mf = new Float64Array(N_MFCC);
    for (let c = 0; c < N_MFCC; c++) { let acc = 0; const row = dctM[c];
      for (let m = 0; m < N_MEL; m++) acc += row[m] * mel[m]; mf[c] = acc; }
    mfccFrames.push(mf);
    cent.push(centroid); band.push(bw); roll.push(rolloff);
    zcr.push(zc / WIN); rms.push(Math.sqrt(energy / WIN));
  }
  if (mfccFrames.length === 0) mfccFrames.push(new Float64Array(N_MFCC));

  // MFCC 평균/표준편차
  const feats = [];
  for (let c = 0; c < N_MFCC; c++) {
    const col = mfccFrames.map((f) => f[c]); feats.push(...stats(col));
  }
  // delta MFCC
  for (let c = 0; c < N_MFCC; c++) {
    const d = [];
    for (let t = 1; t < mfccFrames.length; t++) d.push(mfccFrames[t][c] - mfccFrames[t - 1][c]);
    feats.push(...stats(d.length ? d : [0]));
  }
  for (const arr of [cent, band, roll, zcr, rms]) feats.push(...stats(arr.length ? arr : [0]));
  feats.push(y.length / TARGET_SR);
  return Float64Array.from(feats);
}
