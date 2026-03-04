/**
 * Cross-platform content protection helper.
 *
 * - **Windows 10 2004+**: Uses `SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)`
 *   (0x11) via PowerShell P/Invoke — window is **completely invisible** in captures.
 * - **macOS**: Falls back to Electron's `setContentProtection(true)` which uses
 *   `NSWindow.sharingType = .none` — window shows as a **black rectangle** in captures.
 * - **Linux**: Falls back to Electron's `setContentProtection(true)` — limited or no
 *   effect on most compositors (Wayland/X11).
 */

import { exec } from "node:child_process";
import type { BrowserWindow } from "electron";

const WDA_NONE = 0;
const WDA_EXCLUDEFROMCAPTURE = 0x11;

/**
 * Apply content protection to a BrowserWindow.
 * On Windows 10 2004+ this uses WDA_EXCLUDEFROMCAPTURE (fully invisible).
 * On other platforms it falls back to Electron's built-in method.
 */
export function applyContentProtection(
  win: BrowserWindow,
  enabled: boolean,
): void {
  if (process.platform !== "win32") {
    if (process.platform === "linux" && enabled) {
      console.warn(
        "Content protection has limited support on Linux. " +
        "Window content may still be visible in screen captures.",
      );
    }
    win.setContentProtection(enabled);
    return;
  }

  const hwndBuf = win.getNativeWindowHandle();
  const hwnd =
    hwndBuf.length >= 8
      ? hwndBuf.readBigUInt64LE(0)
      : BigInt(hwndBuf.readUInt32LE(0));
  const affinity = enabled ? WDA_EXCLUDEFROMCAPTURE : WDA_NONE;

  // Build a small PowerShell script that P/Invokes SetWindowDisplayAffinity.
  // Using -EncodedCommand (Base64 of UTF-16LE) avoids all quoting issues.
  const ps = [
    "Add-Type 'using System;using System.Runtime.InteropServices;public class W{[DllImport(\"user32.dll\")]public static extern bool SetWindowDisplayAffinity(IntPtr h,uint a);}'",
    `[W]::SetWindowDisplayAffinity([IntPtr]::new(${hwnd.toString()}),${affinity})`,
  ].join(";");

  const encoded = Buffer.from(ps, "utf16le").toString("base64");

  exec(
    `powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand ${encoded}`,
    { windowsHide: true, timeout: 5000 },
    (err) => {
      if (err) console.error("SetWindowDisplayAffinity failed:", err);
    },
  );
}
