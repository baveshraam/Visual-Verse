import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    base: './',
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
            '@components': path.resolve(__dirname, './src/components'),
            '@hooks': path.resolve(__dirname, './src/hooks'),
        }
    },
    build: {
        outDir: 'dist',
        emptyOutDir: true,
        sourcemap: false,
        rollupOptions: {
            output: {
                manualChunks: {
                    three: ['three', '@react-three/fiber', '@react-three/drei'],
                    vendor: ['react', 'react-dom', 'framer-motion']
                }
            }
        }
    },
    server: {
        port: 5173,
        strictPort: true,
        cors: true
    },
    optimizeDeps: {
        include: ['three', '@react-three/fiber', '@react-three/drei', '@react-three/postprocessing']
    }
})
