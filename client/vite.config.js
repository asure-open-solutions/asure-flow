import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import electron from "vite-plugin-electron";
import electronRenderer from "vite-plugin-electron-renderer";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";
const webOnly = process.env.ASUREFLOW_WEB_ONLY === "1";
export default defineConfig({
    plugins: [
        react(),
        tailwindcss(),
        ...(!webOnly ? electron([
            {
                entry: "electron/main.ts",
                onstart(args) {
                    args.startup();
                },
                vite: {
                    build: {
                        outDir: "dist-electron",
                        rollupOptions: {
                            external: ["electron"],
                            output: {
                                format: "cjs",
                            },
                        },
                    },
                },
            },
            {
                entry: "electron/preload.ts",
                onstart(args) {
                    args.reload();
                },
                vite: {
                    build: {
                        outDir: "dist-electron",
                        rollupOptions: {
                            output: {
                                format: "cjs",
                            },
                        },
                    },
                },
            },
        ]) : []),
        ...(!webOnly ? [electronRenderer()] : []),
    ],
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "src"),
        },
    },
});
