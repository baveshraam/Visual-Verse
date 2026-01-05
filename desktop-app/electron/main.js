import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let mainWindow = null;
let pythonProcess = null;

const isDev = !app.isPackaged;

// Python backend configuration
const PYTHON_PORT = 8000;
const PYTHON_HOST = 'localhost';

function startPythonBackend() {
    const projectRoot = path.join(__dirname, '..', '..');
    const pythonScript = path.join(projectRoot, 'api', 'routes.py');

    // Check if backend is already running
    fetch(`http://${PYTHON_HOST}:${PYTHON_PORT}/health`)
        .then(() => {
            console.log('Python backend already running');
        })
        .catch(() => {
            console.log('Starting Python backend...');

            // Try different Python commands
            const pythonCommands = ['python', 'python3', 'py'];

            for (const pythonCmd of pythonCommands) {
                try {
                    pythonProcess = spawn(pythonCmd, [
                        '-m', 'uvicorn',
                        'api.routes:app',
                        '--host', PYTHON_HOST,
                        '--port', String(PYTHON_PORT),
                        '--reload'
                    ], {
                        cwd: projectRoot,
                        shell: true,
                        stdio: ['pipe', 'pipe', 'pipe']
                    });

                    pythonProcess.stdout.on('data', (data) => {
                        console.log(`Python: ${data}`);
                    });

                    pythonProcess.stderr.on('data', (data) => {
                        console.error(`Python Error: ${data}`);
                    });

                    pythonProcess.on('error', (err) => {
                        console.error('Failed to start Python backend:', err);
                    });

                    pythonProcess.on('close', (code) => {
                        console.log(`Python backend exited with code ${code}`);
                    });

                    console.log(`Started Python backend with: ${pythonCmd}`);
                    break;
                } catch (err) {
                    console.log(`Failed to start with ${pythonCmd}, trying next...`);
                }
            }
        });
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1600,
        height: 1000,
        minWidth: 1200,
        minHeight: 800,
        frame: true, // Use standard frame for resize/minimize/maximize
        resizable: true,
        backgroundColor: '#0a0a0f',
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js'),
            webSecurity: true
        },
        show: false, // Don't show until ready
        icon: path.join(__dirname, '..', 'public', 'icon.png')
    });

    // Load the app
    if (isDev) {
        mainWindow.loadURL('http://localhost:5173');
        mainWindow.webContents.openDevTools();
    } else {
        mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
    }

    // Show window when ready
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
        mainWindow.focus();
    });

    // Handle window close
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

// IPC handlers for window controls
ipcMain.on('window-minimize', () => {
    if (mainWindow) mainWindow.minimize();
});

ipcMain.on('window-maximize', () => {
    if (mainWindow) {
        if (mainWindow.isMaximized()) {
            mainWindow.unmaximize();
        } else {
            mainWindow.maximize();
        }
    }
});

ipcMain.on('window-close', () => {
    if (mainWindow) mainWindow.close();
});

ipcMain.handle('get-backend-url', () => {
    return `http://${PYTHON_HOST}:${PYTHON_PORT}`;
});

// App lifecycle
app.whenReady().then(() => {
    startPythonBackend();

    // Wait a bit for backend to start
    setTimeout(() => {
        createWindow();
    }, 2000);
});

app.on('window-all-closed', () => {
    // Kill Python backend
    if (pythonProcess) {
        pythonProcess.kill();
    }

    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});

app.on('before-quit', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
});
