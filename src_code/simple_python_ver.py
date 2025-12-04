import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import map_coordinates
from matplotlib.path import Path

# --- 1. Fluid Engine (Physically Based) ---

class FluidSimulation:
    def __init__(self, size=512, dt=0.01):
        # GRID SETTINGS
        self.size = size  # N (grid resolution)
        self.domain_length = 1.0 # L (Physical length in meters)
        self.dx = self.domain_length / self.size # dx (Cell size in meters)
        
        # PHYSICS SETTINGS
        # dt is time step in seconds
        self.dt = dt  
        # diff is Kinematic Viscosity (m^2/s). 
        # Water is approx 1.0e-6 m^2/s.
        self.diff = 0.0 
        self.visc = 0.0

        # Grid setup
        self.x = np.arange(size)
        self.y = np.arange(size)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Physics Fields
        self.density = np.zeros((size, size))
        # Velocities are stored in m/s
        self.Vx = np.zeros((size, size)) 
        self.Vy = np.zeros((size, size)) 

        self.Vx0 = np.zeros((size, size))
        self.Vy0 = np.zeros((size, size))
        self.s = np.zeros((size, size))

        # Interaction
        self.mouse_down = False
        self.mouse_pos = (0, 0)
        self.prev_mouse_pos = (0, 0)

        # Simulation State
        self.mode = "interactive"
        self.inflow_velocity = 0.0 # in m/s
        self.obstacle_mask = np.zeros((size, size), dtype=bool)
        
        # Geometry size parameter (visual size in grid units)
        self.geo_size = 20.0 

    def set_obstacle(self, mode, size_param, m=0.02, p=0.4, t=0.12, angle=0.0):
        self.obstacle_mask[:] = False
        self.mode = mode
        self.geo_size = size_param

        cx, cy = self.size // 2, self.size // 2

        if mode == "sphere":
            radius = size_param
            mask = ((self.X - cx)**2 + (self.Y - cy)**2) < radius**2
            self.obstacle_mask[mask] = True

        elif mode == "aero":
            # Generate a NACA 4-digit airfoil
            chord = max(4.0, size_param * 3.0)
            
            # NACA Parameters
            m = float(np.clip(m, 0.0, 0.1))   # Max camber
            p = float(np.clip(p, 0.05, 0.95)) # Max camber position
            t = float(np.clip(t, 0.02, 0.30)) # Thickness
            angle_deg = float(angle)
            angle_rad = np.deg2rad(angle_deg)
            cosA = np.cos(angle_rad)
            sinA = np.sin(angle_rad)

            num = max(200, int(chord * 6))
            x_rel = np.linspace(0.0, 1.0, num)

            # NACA 4-digit Thickness distribution
            yt = 5 * t * (0.2969 * np.sqrt(np.maximum(x_rel, 0.0)) - 0.1260 * x_rel - 0.3516 * x_rel**2
                          + 0.2843 * x_rel**3 - 0.1015 * x_rel**4)

            # Camber line calculation
            yc = np.zeros_like(x_rel)
            dyc_dx = np.zeros_like(x_rel)
            if p > 0 and m > 0:
                mask1 = x_rel < p
                mask2 = ~mask1
                # First region (0 to p)
                yc[mask1] = (m / p**2) * (2 * p * x_rel[mask1] - x_rel[mask1]**2)
                dyc_dx[mask1] = (2 * m / p**2) * (p - x_rel[mask1])
                # Second region (p to 1)
                yc[mask2] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x_rel[mask2] - x_rel[mask2]**2)
                dyc_dx[mask2] = (2 * m / (1 - p)**2) * (p - x_rel[mask2])

            # Convert to absolute coordinates
            x_abs = (cx - chord / 2.0) + x_rel * chord
            yc_abs = cy + yc * chord
            yt_abs = yt * chord

            theta = np.arctan(dyc_dx)

            # Upper and Lower surface
            xu = x_abs - yt_abs * np.sin(theta)
            yu = yc_abs + yt_abs * np.cos(theta)
            xl = x_abs + yt_abs * np.sin(theta)
            yl = yc_abs - yt_abs * np.cos(theta)

            upper = np.column_stack([xu, yu])
            lower = np.column_stack([xl, yl])
            polygon = np.vstack([upper, lower[::-1]])

            # Rotate
            poly_shifted = polygon - np.array([cx, cy])[None, :]
            rot_matrix = np.array([[cosA, -sinA], [sinA, cosA]])
            polygon_rot = (poly_shifted @ rot_matrix.T) + np.array([cx, cy])[None, :]

            # Clip to grid
            polygon_rot[:, 0] = np.clip(polygon_rot[:, 0], 0.0, self.size - 1.0)
            polygon_rot[:, 1] = np.clip(polygon_rot[:, 1], 0.0, self.size - 1.0)

            # Rasterize
            path = Path(polygon_rot)
            points = np.vstack((self.X.ravel(), self.Y.ravel())).T
            mask_flat = path.contains_points(points)
            mask = mask_flat.reshape(self.size, self.size)

            self.obstacle_mask[mask] = True

        elif mode == "triangles":
            step = 30
            t_size = 5
            for r in range(20, self.size - 20, step):
                for c in range(40, self.size - 20, step):
                    dist = np.abs(self.X - c) + np.abs(self.Y - r)
                    mask = dist < t_size
                    self.obstacle_mask[mask] = True

        if np.any(self.obstacle_mask):
            self.density[self.obstacle_mask] = 0
            self.Vx[self.obstacle_mask] = 0
            self.Vy[self.obstacle_mask] = 0

    def add_density(self, x, y, amount=100):
        radius = 5
        r_mask = ((self.X - x)**2 + (self.Y - y)**2) < radius**2
        r_mask = np.logical_and(r_mask, ~self.obstacle_mask)
        self.density[r_mask] += amount
        self.density = np.clip(self.density, 0, 255)

    def add_velocity(self, x, y, amount_x, amount_y):
        radius = 5
        r_mask = ((self.X - x)**2 + (self.Y - y)**2) < radius**2
        r_mask = np.logical_and(r_mask, ~self.obstacle_mask)
        self.Vx[r_mask] += amount_x
        self.Vy[r_mask] += amount_y

    def diffuse(self, b, x, x0, diff_coeff):
        # Physics: alpha = (dt * viscosity) / dx^2
        # Since domain L=1, dx = 1/N. Thus 1/dx^2 = N^2.
        a = self.dt * diff_coeff * (self.size - 2) * (self.size - 2)
        
        # Jacobi Iteration
        for _ in range(5):
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (
                x[:-2, 1:-1] + x[2:, 1:-1] +
                x[1:-1, :-2] + x[1:-1, 2:]
            )) / (1 + 4 * a)

    def project(self, velocX, velocY, p, div):
        n = self.size
        div[1:-1, 1:-1] = -0.5 * (
            (velocX[1:-1, 2:] - velocX[1:-1, :-2]) +
            (velocY[2:, 1:-1] - velocY[:-2, 1:-1])
        ) / n

        p[:] = 0
        for _ in range(10):
            p[1:-1, 1:-1] = (div[1:-1, 1:-1] +
                             p[:-2, 1:-1] + p[2:, 1:-1] +
                             p[1:-1, :-2] + p[1:-1, 2:]) / 4

        velocX[1:-1, 1:-1] -= 0.5 * n * (p[1:-1, 2:] - p[1:-1, :-2])
        velocY[1:-1, 1:-1] -= 0.5 * n * (p[2:, 1:-1] - p[:-2, 1:-1])

    def advect(self, b, d, d0, velocX, velocY):
        dt0 = self.dt * self.size
        i, j = np.indices((self.size, self.size))

        row_pos = i - dt0 * velocY
        col_pos = j - dt0 * velocX

        row_pos = np.clip(row_pos, 0.5, self.size - 1.5)
        col_pos = np.clip(col_pos, 0.5, self.size - 1.5)

        d[:] = map_coordinates(d0, [row_pos, col_pos], order=1, mode='nearest')

    def step(self):
        # 1. Apply Inflow
        if self.mode != "interactive":
            mid_start = int(self.size * 0.3)
            mid_end = int(self.size * 0.7)
            self.Vx[mid_start:mid_end, 0:5] = self.inflow_velocity
            self.Vy[mid_start:mid_end, 0:5] = 0
            self.density[mid_start:mid_end, 0:5] = 200

        # Mouse Interaction
        if self.mouse_down:
            mx, my = self.mouse_pos
            px, py = self.prev_mouse_pos
            self.add_density(mx, my, amount=50)
            
            force_scale = 50.0 
            force_x = (mx - px) * force_scale * self.dx 
            force_y = (my - py) * force_scale * self.dx
            
            self.add_velocity(mx, my, force_x, force_y)
            self.prev_mouse_pos = (mx, my)

        # 2. Viscosity
        if self.visc > 0:
            self.Vx0[:] = self.Vx[:]
            self.Vy0[:] = self.Vy[:]
            self.diffuse(1, self.Vx, self.Vx0, self.visc)
            self.diffuse(2, self.Vy, self.Vy0, self.visc)

        # 3. Project
        self.project(self.Vx, self.Vy, self.Vx0, self.Vy0)

        # 4. Advect
        self.Vx0[:] = self.Vx[:]
        self.Vy0[:] = self.Vy[:]
        self.advect(1, self.Vx, self.Vx0, self.Vx0, self.Vy0)
        self.advect(2, self.Vy, self.Vy0, self.Vx0, self.Vy0)

        self.s[:] = self.density[:]
        self.advect(0, self.density, self.s, self.Vx, self.Vy)

        # 5. Project again
        self.project(self.Vx, self.Vy, self.Vx0, self.Vy0)

        # 6. Obstacles
        if np.any(self.obstacle_mask):
            self.Vx[self.obstacle_mask] = 0
            self.Vy[self.obstacle_mask] = 0
            self.density[self.obstacle_mask] = 0

        self.density *= 0.995

# --- 2. GUI Application ---

class FluidApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("1m x 1m Physical Fluid Simulator")
        self.geometry("1100x850") # Increased height for extra controls

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.running = False

        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.frames = {}
        # Domain 1m, dt = 10ms
        self.fluid = FluidSimulation(size=256, dt=0.01) 

        self.obstacle_color = (1.0, 0.0, 0.0, 1.0)
        self.current_mode = "interactive"

        self.init_main_menu()
        self.init_selection_menu()
        self.init_simulation_screen()

        self.show_frame("MainMenu")

    def on_closing(self):
        self.running = False
        self.quit()
        self.destroy()

    def show_frame(self, name):
        for frame in self.frames.values():
            frame.pack_forget()
        self.frames[name].pack(fill="both", expand=True)

        if name == "Simulation":
            self.start_simulation()
        else:
            self.stop_simulation()

    def init_main_menu(self):
        frame = tk.Frame(self.container)
        self.frames["MainMenu"] = frame
        tk.Label(frame, text="2D Physical Fluid Solver", font=("Arial", 24, "bold")).pack(pady=50)
        tk.Label(frame, text="Domain: 1m x 1m | Grid: 256x256", font=("Arial", 12)).pack(pady=5)
        
        btn_start = tk.Button(frame, text="Start", font=("Arial", 14), width=15,
                              command=lambda: self.show_frame("SelectionMenu"))
        btn_start.pack(pady=20)
        btn_quit = tk.Button(frame, text="Quit", font=("Arial", 14), width=15,
                             command=self.on_closing)
        btn_quit.pack(pady=10)

    def init_selection_menu(self):
        frame = tk.Frame(self.container)
        self.frames["SelectionMenu"] = frame
        tk.Label(frame, text="Select Flow Case", font=("Arial", 18)).pack(pady=30)
        options = [
            ("1. Mouse Interaction (Still Water)", "interactive"),
            ("2. Sphere / Cylinder Flow", "sphere"),
            ("3. Airfoil (NACA) Flow", "aero"),
            ("4. Porous Media (Triangles)", "triangles")
        ]
        for text, mode in options:
            btn = tk.Button(frame, text=text, font=("Arial", 12), width=40,
                            command=lambda m=mode: self.launch_simulation(m))
            btn.pack(pady=5)
        tk.Button(frame, text="Back", font=("Arial", 12), width=20,
                  command=lambda: self.show_frame("MainMenu")).pack(pady=30)

    def init_simulation_screen(self):
        frame = tk.Frame(self.container)
        self.frames["Simulation"] = frame

        control_panel = tk.Frame(frame, width=320, bg="#f0f0f0")
        control_panel.pack(side="left", fill="y", padx=5, pady=5)

        canvas_panel = tk.Frame(frame)
        canvas_panel.pack(side="right", fill="both", expand=True)

        # --- Physical Controls ---
        tk.Label(control_panel, text="Physical Parameters", bg="#f0f0f0", font=("Arial", 12, "bold")).pack(pady=10)

        # Velocity Slider (m/s)
        self.var_velocity = tk.DoubleVar(value=0.0)
        tk.Label(control_panel, text="Inflow Velocity (m/s)", bg="#f0f0f0").pack(pady=(5,0))
        self.scale_vel = tk.Scale(control_panel, variable=self.var_velocity, from_=0.0, to=5.0,
                                  orient="horizontal", length=280, resolution=0.1)
        self.scale_vel.pack()

        # Viscosity Slider (Scaled to Water)
        self.var_visc = tk.DoubleVar(value=0.0)
        tk.Label(control_panel, text="Kinematic Viscosity (m^2/s)", bg="#f0f0f0").pack(pady=(5,0))
        self.scale_visc = tk.Scale(control_panel, variable=self.var_visc, from_=0.0, to=0.000001,
                                   orient="horizontal", length=280, resolution=0.00000001)
        self.scale_visc.pack()

        # Geometry Size
        self.var_size = tk.DoubleVar(value=20.0)
        self.lbl_size = tk.Label(control_panel, text="Obstacle Size (Grid Units)", bg="#f0f0f0")
        self.lbl_size.pack(pady=(10,0))
        self.scale_size = tk.Scale(control_panel, variable=self.var_size, from_=10.0, to=80.0,
                                   orient="horizontal", length=280, resolution=1,
                                   command=self.update_geometry)
        self.scale_size.pack()
        self.lbl_cm = tk.Label(control_panel, text="Approx: 7.8 cm", bg="#f0f0f0", fg="blue")
        self.lbl_cm.pack()

        # --- Aero Controls ---
        tk.Label(control_panel, text="Airfoil Parameters", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(pady=(15,0))
        
        tk.Label(control_panel, text="Thickness t (%)", bg="#f0f0f0").pack()
        self.var_t = tk.DoubleVar(value=0.12)
        self.scale_t = tk.Scale(control_panel, variable=self.var_t, from_=0.02, to=0.25,
                                orient="horizontal", length=280, resolution=0.01, command=self.update_geometry)
        self.scale_t.pack()

        # RESTORED: Camber M
        tk.Label(control_panel, text="Camber m (Curvature)", bg="#f0f0f0").pack()
        self.var_m = tk.DoubleVar(value=0.02)
        self.scale_m = tk.Scale(control_panel, variable=self.var_m, from_=0.0, to=0.1,
                                orient="horizontal", length=280, resolution=0.005, command=self.update_geometry)
        self.scale_m.pack()

        # RESTORED: Position P
        tk.Label(control_panel, text="Camber Pos p (Peak Location)", bg="#f0f0f0").pack()
        self.var_p = tk.DoubleVar(value=0.4)
        self.scale_p = tk.Scale(control_panel, variable=self.var_p, from_=0.1, to=0.9,
                                orient="horizontal", length=280, resolution=0.05, command=self.update_geometry)
        self.scale_p.pack()

        tk.Label(control_panel, text="Angle (deg)", bg="#f0f0f0").pack()
        self.var_angle = tk.DoubleVar(value=0.0)
        self.scale_angle = tk.Scale(control_panel, variable=self.var_angle, from_=-20, to=20,
                                    orient="horizontal", length=280, resolution=1, command=self.update_geometry)
        self.scale_angle.pack()

        tk.Button(control_panel, text="Back to Menu", command=self.stop_and_back).pack(pady=20)
        tk.Button(control_panel, text="Clear Fluid", command=self.clear_fluid).pack(pady=5)

        # Canvas
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        rgba = plt.get_cmap('turbo')(plt.Normalize(vmin=0, vmax=255)(self.fluid.density))
        rgba[self.fluid.obstacle_mask] = self.obstacle_color
        self.im = self.ax.imshow(rgba, origin='lower', interpolation='nearest')
        self.ax.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_move)

    def clear_fluid(self):
        self.fluid.density[:] = 0
        self.fluid.Vx[:] = 0
        self.fluid.Vy[:] = 0
        self.fluid.obstacle_mask[:] = False
        self.update_geometry() # redraw obstacle

    def launch_simulation(self, mode):
        self.fluid = FluidSimulation(size=256, dt=0.01)
        self.current_mode = mode

        def set_state(widget, state): widget.config(state=state)

        if mode == "interactive":
            set_state(self.scale_vel, "disabled")
            set_state(self.scale_size, "disabled")
            set_state(self.scale_t, "disabled")
            set_state(self.scale_m, "disabled")
            set_state(self.scale_p, "disabled")
            set_state(self.scale_angle, "disabled")
            self.var_velocity.set(0)
        elif mode == "triangles":
            set_state(self.scale_vel, "normal")
            set_state(self.scale_size, "disabled")
            set_state(self.scale_t, "disabled")
            set_state(self.scale_m, "disabled")
            set_state(self.scale_p, "disabled")
            set_state(self.scale_angle, "disabled")
            self.var_velocity.set(1.0) 
        else: # sphere / aero
            set_state(self.scale_vel, "normal")
            set_state(self.scale_size, "normal")
            # Only enable aero params if in aero mode
            is_aero = "normal" if mode == "aero" else "disabled"
            set_state(self.scale_t, is_aero)
            set_state(self.scale_m, is_aero)
            set_state(self.scale_p, is_aero)
            set_state(self.scale_angle, is_aero)
            
            self.var_velocity.set(1.0)
            self.var_size.set(20.0)

        self.update_geometry()
        self.show_frame("Simulation")

    def update_geometry(self, val=None):
        grid_units = self.var_size.get()
        cm_size = (grid_units / 256.0) * 100.0
        self.lbl_cm.config(text=f"Approx Size: {cm_size:.1f} cm")

        if getattr(self, 'current_mode', None) in ["sphere", "aero", "triangles"]:
            if self.current_mode == "aero":
                self.fluid.set_obstacle("aero", float(self.var_size.get()),
                                         m=self.var_m.get(), p=self.var_p.get(), 
                                         t=self.var_t.get(), angle=self.var_angle.get())
            else:
                self.fluid.set_obstacle(self.current_mode, float(self.var_size.get()))

            if hasattr(self, 'im'):
                rgba = plt.get_cmap('turbo')(plt.Normalize(vmin=0, vmax=255)(self.fluid.density))
                rgba[self.fluid.obstacle_mask] = self.obstacle_color
                self.im.set_data(rgba)
                self.canvas.draw_idle()

    def start_simulation(self):
        self.running = True
        self.animate_loop()

    def stop_simulation(self):
        self.running = False

    def stop_and_back(self):
        self.stop_simulation()
        self.show_frame("SelectionMenu")

    def animate_loop(self):
        if not self.running: return

        self.fluid.inflow_velocity = self.var_velocity.get()
        self.fluid.visc = self.var_visc.get()

        self.fluid.step()

        display_data = self.fluid.density.copy()
        if np.any(self.fluid.obstacle_mask):
            display_data[self.fluid.obstacle_mask] = 0

        rgba = plt.get_cmap('turbo')(plt.Normalize(vmin=0, vmax=255)(display_data))
        if np.any(self.fluid.obstacle_mask):
            rgba[self.fluid.obstacle_mask] = self.obstacle_color

        self.im.set_data(rgba)
        self.canvas.draw()

        if self.running:
            self.after(1, self.animate_loop)

    def on_click(self, event):
        if event.inaxes != self.ax: return
        self.fluid.mouse_down = True
        self.fluid.mouse_pos = (event.xdata, event.ydata)
        self.fluid.prev_mouse_pos = (event.xdata, event.ydata)

    def on_release(self, event):
        self.fluid.mouse_down = False

    def on_move(self, event):
        if event.inaxes != self.ax: return
        self.fluid.mouse_pos = (event.xdata, event.ydata)

if __name__ == "__main__":
    app = FluidApp()
    app.mainloop()