/**
 * ComfyUI SAM3DBody - Interactive Viewport with Safe Frame
 * Reliability Update: Promise-based loading & Instant Render
 */

import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

// --- Helper: Promise-based Script Loader ---
function loadScript(src) {
    return new Promise((resolve, reject) => {
        // If already loaded, resolve immediately
        if (document.querySelector(`script[src="${src}"]`)) {
            resolve();
            return;
        }
        const script = document.createElement("script");
        script.src = src;
        script.async = true;
        script.onload = resolve;
        script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
        document.head.appendChild(script);
    });
}

// Global Listener for Mesh Data
api.addEventListener("sam3d_mesh_update", (event) => {
    const node = app.graph.getNodeById(event.detail.node_id);
    if (node) {
        // If Three.js is ready, update immediately
        if (node.updateMesh && node.isThreeInitialized) {
            node.updateMesh(event.detail);
        } else {
            // Otherwise cache it for the init routine
            node.cachedMeshData = event.detail;
        }
    }
});

app.registerExtension({
    name: "Comfy.SAM3DVisualizer.Interactive.DOM",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SAM3DBodyVisualize") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.isThreeInitialized = false;

                // 1. Create Main Container
                const container = document.createElement("div");
                Object.assign(container.style, {
                    width: "100%", 
                    height: "100%", 
                    backgroundColor: "#1a1a1a",
                    position: "relative", 
                    pointerEvents: "auto", 
                    overflow: "hidden"
                });

                // 2. Create Safe Frame
                const safeFrame = document.createElement("div");
                Object.assign(safeFrame.style, {
                    position: "absolute", 
                    top: "50%", 
                    left: "50%",
                    transform: "translate(-50%, -50%)",
                    border: "2px dashed rgba(255, 200, 0, 0.9)", 
                    boxShadow: "0 0 0 1000px rgba(0, 0, 0, 0.6)", 
                    pointerEvents: "none", 
                    zIndex: "50",
                    display: "none", 
                    boxSizing: "border-box"
                });
                container.appendChild(safeFrame);
                this.safeFrameEl = safeFrame;

                // 3. Reset Button
                const resetBtn = document.createElement("button");
                resetBtn.innerText = "↺ Reset View";
                Object.assign(resetBtn.style, {
                    position: "absolute", top: "10px", right: "10px", zIndex: "100",
                    background: "rgba(0,0,0,0.6)", color: "white", border: "1px solid #555",
                    cursor: "pointer", padding: "5px 10px", borderRadius: "4px", fontSize: "12px"
                });
                
                resetBtn.onclick = () => {
                    const widget = this.widgets.find(w => w.name === "camera_info");
                    if (widget) {
                        widget.value = "";
                        app.graph.setDirtyCanvas(true);
                    }
                    if (this.initialData) this.applyCameraSettings(this.initialData);
                };
                container.appendChild(resetBtn);

                // 4. Add Widget
                const widget = this.addDOMWidget("viewport", "3D_VIEW", container, {
                    getValue() { return ""; },
                    setValue(v) { },
                });
                widget.computeSize = (w) => [w, 350]; 
                this.setSize([this.size[0], this.size[1] + 350]);

                // 5. Robust Initialization Sequence
                this.initializeDependencies(container).then(() => {
                    this.isThreeInitialized = true;
                    // Check if data arrived while we were loading
                    if (this.cachedMeshData) {
                        this.updateMesh(this.cachedMeshData);
                        this.cachedMeshData = null;
                    }
                }).catch(err => console.error("SAM3D: Failed to load 3D libraries", err));

                return r;
            };

            // New method to handle async loading reliably
            nodeType.prototype.initializeDependencies = async function(container) {
                // Load Three.js core
                if (!window.THREE) {
                    await loadScript("https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js");
                }
                // Load OrbitControls (dependent on THREE)
                if (!window.THREE.OrbitControls) {
                    await loadScript("https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js");
                }
                // Initialize Scene
                this.initThree(container);
            };

            nodeType.prototype.initThree = function(container) {
                this.container = container;
                this.scene = new THREE.Scene();
                this.scene.background = new THREE.Color(0x222222);
                
                // Camera
                const aspect = container.clientWidth / 350;
                this.camera = new THREE.PerspectiveCamera(30, aspect, 0.1, 1000);
                
                // Renderer
                this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
                this.renderer.setSize(container.clientWidth, 350);
                container.appendChild(this.renderer.domElement);
                
                // Resize Handler
                new ResizeObserver(() => {
                    if (container.clientWidth > 0) {
                        const w = container.clientWidth;
                        const h = 350;
                        
                        this.camera.aspect = w / h;
                        this.camera.updateProjectionMatrix();
                        this.renderer.setSize(w, h);
                        
                        if(this.initialData) {
                            this.updateSafeFrame(this.initialData.width, this.initialData.height, w, h);
                        }
                        // Force redraw on resize
                        this.renderer.render(this.scene, this.camera);
                    }
                }).observe(container);

                // Lighting
                const light = new THREE.DirectionalLight(0xffffff, 1);
                light.position.set(0, 1, 2);
                this.scene.add(light);
                this.scene.add(new THREE.AmbientLight(0x555555));
                
                // Controls
                this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.1;
                this.controls.addEventListener('change', () => this.syncCameraToWidget());
                
                this.animate();
            };

            nodeType.prototype.animate = function() {
                requestAnimationFrame(() => this.animate());
                if(this.controls) this.controls.update();
                if(this.renderer) this.renderer.render(this.scene, this.camera);
            };

            nodeType.prototype.updateMesh = function(data) {
                this.initialData = data; 
                
                if (this.container) {
                    this.updateSafeFrame(data.width, data.height, this.container.clientWidth, 350);
                }

                if (!this.scene) return;
                
                if (this.currentMesh) {
                    this.scene.remove(this.currentMesh);
                    this.currentMesh.geometry.dispose();
                }

                const geometry = new THREE.BufferGeometry();
                const vertices = new Float32Array(data.vertices);
                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                geometry.setIndex(data.faces);
                geometry.computeVertexNormals();
                
                const material = new THREE.MeshPhongMaterial({ 
                    color: 0x00ffcc, 
                    flatShading: true, 
                    side: THREE.DoubleSide 
                });
                
                this.currentMesh = new THREE.Mesh(geometry, material);
                this.currentMesh.rotation.x = Math.PI; 
                this.scene.add(this.currentMesh);
                
                // Initial Camera Position
                const widget = this.widgets.find(w => w.name === "camera_info");
                if (!widget || !widget.value) this.applyCameraSettings(data);

                // --- INSTANT UPDATE FORCE ---
                // Render immediately to prevent "blank" frame waiting for animation loop
                this.renderer.render(this.scene, this.camera);
                // Mark ComfyUI canvas as dirty to ensure the UI updates visually
                app.graph.setDirtyCanvas(true, true);
            };

            nodeType.prototype.updateSafeFrame = function(imgW, imgH, viewW, viewH) {
                if(!this.safeFrameEl || !imgW || !imgH) return;
                
                const targetAspect = imgW / imgH;
                const viewAspect = viewW / viewH;
                
                let renderW, renderH;
                if (viewAspect > targetAspect) {
                    renderH = viewH;
                    renderW = viewH * targetAspect;
                } else {
                    renderW = viewW;
                    renderH = viewW / targetAspect;
                }
                
                Object.assign(this.safeFrameEl.style, {
                    width: `${renderW}px`,
                    height: `${renderH}px`,
                    display: "block"
                });
            };

            nodeType.prototype.applyCameraSettings = function(data) {
                const pos = data.cam_pos;
                // INVERTED X-AXIS as requested
                const startX = -pos[0]; 
                const startY = pos[1];
                const startZ = pos[2];
                
                this.camera.position.set(startX, startY, startZ);
                this.controls.target.set(startX, startY, 0); 
                this.camera.lookAt(startX, startY, 0);

                const h = data.height;
                const f = data.focal_length;
                if(f > 0) {
                    const fovDeg = 2 * Math.atan(h / (2 * f)) * (180 / Math.PI);
                    this.camera.fov = fovDeg;
                    this.camera.updateProjectionMatrix();
                }
                this.controls.update();
            };

            nodeType.prototype.syncCameraToWidget = function() {
                const widget = this.widgets.find(w => w.name === "camera_info");
                if (!widget) return;
                
                const state = {
                    position: {
                        x: -this.camera.position.x, 
                        y: this.camera.position.y, 
                        z: this.camera.position.z
                    },
                    target: {
                        x: -this.controls.target.x,
                        y: this.controls.target.y,
                        z: this.controls.target.z
                    },
                    fov: this.camera.fov
                };
                widget.value = JSON.stringify(state);
            };
        }
    }
});