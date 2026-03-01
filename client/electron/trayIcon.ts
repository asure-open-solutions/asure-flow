import { nativeImage, NativeImage } from "electron";

/**
 * Rasterizes the Asuré "A" mark SVG path to a bitmap icon at any size.
 * White on transparent, suitable for system trays and window icons.
 */

type Pt = [number, number];

/** Evaluate a cubic bezier curve at parameter t ∈ [0, 1]. */
function cubicAt(p0: Pt, p1: Pt, p2: Pt, p3: Pt, t: number): Pt {
  const u = 1 - t;
  return [
    u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0],
    u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1],
  ];
}

/**
 * Build the outline polygon of the Asuré "A" mark by evaluating
 * the SVG path's cubic bezier curves and line segments.
 */
function buildOutline(steps = 24): Pt[] {
  const pts: Pt[] = [];

  const cubic = (p0: Pt, cp1: Pt, cp2: Pt, p3: Pt) => {
    for (let i = 1; i <= steps; i++) pts.push(cubicAt(p0, cp1, cp2, p3, i / steps));
  };
  const line = (to: Pt) => { pts.push(to); };

  // M 111.525 68.223
  pts.push([111.525, 68.223]);

  cubic([111.525, 68.223], [106.457, 77.701], [88.985, 108.043], [75.799, 130.268]);
  cubic([75.799, 130.268], [61.222, 154.835], [60, 157.357], [60, 162.879]);
  cubic([60, 162.879], [60, 166.034], [60.601, 167.328], [63.005, 169.351]);
  cubic([63.005, 169.351], [67.129, 172.821], [70.919, 173.192], [98.5, 172.822]);
  cubic([98.5, 172.822], [119.341, 172.542], [122.934, 172.263], [125.801, 170.698]);
  cubic([125.801, 170.698], [127.617, 169.706], [129.362, 168.221], [129.678, 167.397]);
  cubic([129.678, 167.397], [130.911, 164.185], [140.147, 149.512], [140.804, 149.723]);
  cubic([140.804, 149.723], [141.187, 149.845], [143.75, 154.327], [146.5, 159.682]);
  cubic([146.5, 159.682], [152.578, 171.519], [154.375, 172.813], [165, 173.009]);
  cubic([165, 173.009], [178.688, 173.26], [178.078, 173.407], [176.702, 170.193]);
  cubic([176.702, 170.193], [176.041, 168.65], [171.865, 161.337], [167.423, 153.943]);
  cubic([167.423, 153.943], [162.981, 146.549], [156.791, 136.143], [153.668, 130.819]);
  cubic([153.668, 130.819], [147.475, 120.259], [144.233, 117], [139.921, 117]);
  cubic([139.921, 117], [135.749, 117], [132.829, 120.261], [125.591, 133]);
  cubic([125.591, 133], [121.998, 139.325], [117.577, 146.75], [115.768, 149.5]);
  line([112.478, 154.5]);
  line([100.489, 155.23]);
  cubic([100.489, 155.23], [93.895, 155.632], [87.922, 155.753], [87.215, 155.498]);
  cubic([87.215, 155.498], [85.517, 154.887], [86.049, 153.888], [103.99, 123.977]);
  cubic([103.99, 123.977], [112.08, 110.49], [120.589, 96.011], [122.898, 91.803]);
  line([127.098, 84.152]);
  line([121.454, 74.326]);
  cubic([121.454, 74.326], [118.351, 68.922], [115.482, 64.151], [115.079, 63.723]);
  cubic([115.079, 63.723], [114.676, 63.296], [113.077, 65.321], [111.525, 68.223]);

  return pts;
}

/** Ray-casting point-in-polygon test. */
function insidePoly(x: number, y: number, poly: Pt[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

/** Rasterize the A mark to an RGBA bitmap at the given pixel size. */
function rasterize(size: number): Buffer {
  const outline = buildOutline(size >= 64 ? 40 : 24);
  const buf = Buffer.alloc(size * size * 4, 0);

  // Bounding box of the SVG shape (with a little padding)
  const minX = 57, maxX = 181, minY = 61, maxY = 176;
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const span = Math.max(maxX - minX, maxY - minY) * 1.12; // 12% padding

  // 2x2 super-sampling offsets for anti-aliasing
  const aa = [0.25, 0.75];

  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      let hits = 0;
      for (const dy of aa) {
        for (const dx of aa) {
          const sx = cx + ((px + dx) / size - 0.5) * span;
          const sy = cy + ((py + dy) / size - 0.5) * span;
          if (insidePoly(sx, sy, outline)) hits++;
        }
      }
      if (hits > 0) {
        const alpha = Math.round((hits / 4) * 255);
        const i = (py * size + px) * 4;
        buf[i] = 255;         // R
        buf[i + 1] = 255;     // G
        buf[i + 2] = 255;     // B
        buf[i + 3] = alpha;   // A (anti-aliased)
      }
    }
  }
  return buf;
}

// Cache the outline since it's constant
let _outline: Pt[] | null = null;

export function createTrayIcon(): NativeImage {
  return nativeImage.createFromBitmap(rasterize(32), { width: 32, height: 32 });
}

export function createAppIcon(): NativeImage {
  return nativeImage.createFromBitmap(rasterize(256), { width: 256, height: 256 });
}
