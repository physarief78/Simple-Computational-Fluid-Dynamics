using GLMakie
using LinearAlgebra
using Random
using Colors
using Observables
using StaticArrays

# Try to load CUDA only if available; guard imports
const HAS_CUDA = try
    @eval using CUDA
        CUDA.functional()
catch
    false
end

if HAS_CUDA
    @info "CUDA available — GPU mode enabled"
    # optional environment hints (keep if you like)
    ENV["JULIA_CUDA_DEVICE"] = "0"
    ENV["CUDA_VISIBLE_DEVICES"] = "0"
    ENV["GLFW_USE_DISCRETE_GPU"] = "1"
    
    CUDA.device!(0)
    dev = CUDA.device()
    @info "Using GPU: $(CUDA.name(dev))"
    @info "GPU Memory: $(round(CUDA.available_memory() / 1024^3, digits=2)) GB free"

    # Warm up
    test_array = CUDA.zeros(Float32, 512, 512)
    test_array .+= Float32(1.0)
    CUDA.synchronize()
else
    @warn "CUDA not available — falling back to CPU mode"
end

# --- Physical constants (SI units) ---
const RHO_WATER = Float32(1000.0)                # kg/m^3
const MU_WATER = Float32(1.0e-3)                 # Pa·s
const NU_WATER = MU_WATER / RHO_WATER            # kinematic viscosity ≈ 1e-6 m^2/s

# --- Generic FluidSimulation struct that holds either CuArray or Array ---
mutable struct FluidSimulation
    size::Int
    domain_size_m::Float32
    dx::Float32
    dt::Float32
    diff::Float32
    visc::Float32

    density    # either CuArray{Float32,2} or Array{Float32,2}
    Vx
    Vy

    Vx0
    Vy0
    s

    obstacle_mask  # Bool array on GPU or CPU

    mode::String
    geo_size_m::Float32
    inflow_velocity::Float32

    use_cuda::Bool
end

# --- Constructor factory ---
function make_simulation(; use_cuda::Bool=HAS_CUDA, gpu_size::Int=1024, cpu_size::Int=512, domain_size_gpu::Real=1.0, dt::Union{Nothing,Real}=nothing)
    if use_cuda
        N = gpu_size
        domain_size_m_f = Float32(domain_size_gpu)
    else
        # downscale domain by 2x so dx_cpu == dx_gpu (user requested parameters downscaled to match)
        N = cpu_size
        domain_size_m_f = Float32(domain_size_gpu * 0.5)  # half the physical size
    end

    dx = domain_size_m_f / Float32(N)
    U_ref = Float32(1.0)
    dt_default = min(Float32(5e-4), Float32(0.5) * dx / U_ref) |> Float32
    dt_val = dt === nothing ? dt_default : Float32(dt)

    if use_cuda
        # allocate CuArrays lazily if CUDA present
        density = CUDA.zeros(Float32, N, N)
        Vx = CUDA.zeros(Float32, N, N)
        Vy = CUDA.zeros(Float32, N, N)
        Vx0 = CUDA.zeros(Float32, N, N)
        Vy0 = CUDA.zeros(Float32, N, N)
        s = CUDA.zeros(Float32, N, N)
        obstacle_mask = CUDA.zeros(Bool, N, N)
        CUDA.synchronize()
    else
        density = zeros(Float32, N, N)
        Vx = zeros(Float32, N, N)
        Vy = zeros(Float32, N, N)
        Vx0 = zeros(Float32, N, N)
        Vy0 = zeros(Float32, N, N)
        s = zeros(Float32, N, N)
        obstacle_mask = falses(N, N)
    end

    FluidSimulation(
        N,
        domain_size_m_f,
        dx,
        Float32(dt_val),
        Float32(0.0),
        Float32(NU_WATER),
        density, Vx, Vy, Vx0, Vy0, s, obstacle_mask,
        "interactive",
        Float32(domain_size_m_f * Float32(0.15)),
        Float32(0.0),
        use_cuda
    )
end

# ---------------------------
# Utility functions that branch on use_cuda
# ---------------------------

# upload_mask!
function upload_mask!(fluid::FluidSimulation, mask_cpu)
    N = fluid.size

    if mask_cpu === nothing
        mask_array = falses(N, N)
    else
        try
            mask_array = Array(mask_cpu)
        catch e
            @warn "upload_mask!: couldn't convert mask_cpu to Array{Bool,2}, using empty mask. Error: $e"
            mask_array = falses(N, N)
        end
        if size(mask_array) != (N, N)
            @warn "upload_mask!: mask size $(size(mask_array)) != expected ($(N), $(N)). Using empty mask instead."
            mask_array = falses(N, N)
        end
    end

    if fluid.use_cuda
        dmask = CuArray(mask_array)
        copyto!(fluid.obstacle_mask, dmask)
        CUDA.synchronize()
    else
        copyto!(fluid.obstacle_mask, mask_array)
    end
    return
end

# zero_where_mask! - GPU uses kernel, CPU uses simple loop
function zero_where_mask!(fluid::FluidSimulation, arr)
    N = fluid.size
    if fluid.use_cuda
        # use GPU kernel like your original
        function kernel!(arr, mask, N)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            if i >= 1 && i <= N && j >= 1 && j <= N
                if mask[i, j]
                    arr[i, j] = zero(eltype(arr))
                end
            end
            return
        end
        threads = (16, 16)
        blocks = (cld(N, threads[1]), cld(N, threads[2]))
        @cuda blocks=blocks threads=threads kernel!(arr, fluid.obstacle_mask, Int32(N))
        CUDA.synchronize()
    else
        M = fluid.obstacle_mask
        for j in 1:N, i in 1:N
            if M[i, j]
                arr[i, j] = zero(eltype(arr))
            end
        end
    end
end

# --- Add density ---
function add_density!(fluid::FluidSimulation, x::Int, y::Int, amount)
    radius = max(2, round(Int, 40 * (fluid.size >= 1024 ? 1.0 : 0.5))) # smaller radius on CPU run
    N = fluid.size

    if fluid.use_cuda
        function density_kernel!(density, obstacle_mask, x, y, amount, radius, N)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            if i <= N && j <= N
                if !obstacle_mask[i, j] && (i - x)^2 + (j - y)^2 < radius^2
                    density[i, j] += amount
                    density[i, j] = clamp(density[i, j], Float32(0.0), Float32(255.0))
                end
            end
            return
        end
        threads = (16, 16)
        blocks = (cld(N, threads[1]), cld(N, threads[2]))
        @cuda blocks=blocks threads=threads density_kernel!(fluid.density, fluid.obstacle_mask,
                                                          Int32(x), Int32(y), Float32(amount),
                                                          Int32(radius), Int32(N))
    else
        M = fluid.obstacle_mask
        D = fluid.density
        for j in max(1, y-radius):min(N, y+radius), i in max(1, x-radius):min(N, x+radius)
            if !M[i, j] && (i - x)^2 + (j - y)^2 < radius^2
                D[i, j] += Float32(amount)
                D[i, j] = clamp(D[i, j], Float32(0.0), Float32(255.0))
            end
        end
    end
end

# --- Add velocity ---
function add_velocity!(fluid::FluidSimulation, x::Int, y::Int, amount_x, amount_y)
    radius = max(2, round(Int, 20 * (fluid.size >= 1024 ? 1.0 : 0.5)))
    N = fluid.size
    if fluid.use_cuda
        function velocity_kernel!(Vx, Vy, obstacle_mask, x, y, amount_x, amount_y, radius, N)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            if i <= N && j <= N
                if !obstacle_mask[i, j] && (i - x)^2 + (j - y)^2 < radius^2
                    Vx[i, j] += amount_x
                    Vy[i, j] += amount_y
                end
            end
            return
        end
        threads = (16, 16)
        blocks = (cld(N, threads[1]), cld(N, threads[2]))
        @cuda blocks=blocks threads=threads velocity_kernel!(fluid.Vx, fluid.Vy, fluid.obstacle_mask,
                                                             Int32(x), Int32(y), Float32(amount_x), Float32(amount_y),
                                                             Int32(radius), Int32(N))
    else
        M = fluid.obstacle_mask
        Vx = fluid.Vx; Vy = fluid.Vy
        for j in max(1, y-radius):min(N, y+radius), i in max(1, x-radius):min(N, x+radius)
            if !M[i, j] && (i - x)^2 + (j - y)^2 < radius^2
                Vx[i, j] += Float32(amount_x)
                Vy[i, j] += Float32(amount_y)
            end
        end
    end
end

# --- set_bnd! for CPU and a thin wrapper for GPU's kernel-based set_bnd! ---
function set_bnd!(N, b, x, use_cuda::Bool=false)
    if use_cuda
        function boundary_kernel!(x, N, b)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            if i >= 2 && i <= N-1
                x[1, i] = (b == 1) ? -x[2, i] : x[2, i]
                x[N, i] = (b == 1) ? -x[N-1, i] : x[N-1, i]
                x[i, 1] = (b == 2) ? -x[i, 2] : x[i, 2]
                x[i, N] = (b == 2) ? -x[i, N-1] : x[i, N-1]
            end
            if threadIdx().x == 1 && blockIdx().x == 1
                x[1, 1] = Float32(0.5) * (x[2, 1] + x[1, 2])
                x[1, N] = Float32(0.5) * (x[2, N] + x[1, N-1])
                x[N, 1] = Float32(0.5) * (x[N-1, 1] + x[N, 2])
                x[N, N] = Float32(0.5) * (x[N-1, N] + x[N, N-1])
            end
            return
        end
        threads = (256,)
        blocks = (cld(N, threads[1]),)
        @cuda blocks=blocks threads=threads boundary_kernel!(x, Int32(N), Int32(b))
    else
        # CPU version
        for i in 2:N-1
            x[1, i] = (b == 1) ? -x[2, i] : x[2, i]
            x[N, i] = (b == 1) ? -x[N-1, i] : x[N-1, i]
            x[i, 1] = (b == 2) ? -x[i, 2] : x[i, 2]
            x[i, N] = (b == 2) ? -x[i, N-1] : x[i, N-1]
        end
        x[1,1] = Float32(0.5)*(x[2,1]+x[1,2])
        x[1,N] = Float32(0.5)*(x[2,N]+x[1,N-1])
        x[N,1] = Float32(0.5)*(x[N-1,1]+x[N,2])
        x[N,N] = Float32(0.5)*(x[N-1,N]+x[N,N-1])
    end
end

# --- CPU diffuse (Gauss-Seidel) and GPU diffuse wrapper to your @cuda version
function diffuse!(fluid::FluidSimulation, b, x, x0, diff)
    N = fluid.size
    if fluid.use_cuda
        a = fluid.dt * diff * Float32(N - 2) * Float32(N - 2)
        function diffuse_kernel!(x, x0, a, N, b)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            if i >= 2 && i <= N-1 && j >= 2 && j <= N-1
                x[i, j] = (x0[i, j] + a * (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1])) / (Float32(1.0) + Float32(4.0) * a)
            end
            return
        end
        threads = (16, 16)
        blocks = (cld(N, threads[1]), cld(N, threads[2]))
        for _ in 1:5
            @cuda blocks=blocks threads=threads diffuse_kernel!(x, x0, Float32(a), Int32(N), Int32(b))
            set_bnd!(N, b, x, true)
        end
    else
        a = fluid.dt * diff * Float32(N - 2) * Float32(N - 2)
        for _ in 1:5
            for j in 2:N-1, i in 2:N-1
                x[i,j] = (x0[i,j] + a*(x[i-1,j] + x[i+1,j] + x[i,j-1] + x[i,j+1])) / (Float32(1.0) + Float32(4.0)*a)
            end
            set_bnd!(N, b, x, false)
        end
    end
end

# --- CPU project and advect implementations + GPU wrappers ---
function project!(fluid::FluidSimulation, velocX, velocY, p, div)
    N = fluid.size
    if fluid.use_cuda
        function div_kernel!(div, velocX, velocY, N)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            if i >= 2 && i <= N-1 && j >= 2 && j <= N-1
                div[i, j] = -Float32(0.5) * ((velocX[i+1, j] - velocX[i-1, j]) + (velocY[i, j+1] - velocY[i, j-1])) / Float32(N)
                p[i, j] = Float32(0.0)
            end
            return
        end

        function pressure_kernel!(p, div, N)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            if i >= 2 && i <= N-1 && j >= 2 && j <= N-1
                p[i, j] = (div[i, j] + p[i-1, j] + p[i+1, j] + p[i, j-1] + p[i, j+1]) / Float32(4.0)
            end
            return
        end

        function velocity_update_kernel!(velocX, velocY, p, N)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            if i >= 2 && i <= N-1 && j >= 2 && j <= N-1
                velocX[i, j] -= Float32(0.5) * Float32(N) * (p[i+1, j] - p[i-1, j])
                velocY[i, j] -= Float32(0.5) * Float32(N) * (p[i, j+1] - p[i, j-1])
            end
            return
        end

        threads = (16, 16)
        blocks = (cld(N, threads[1]), cld(N, threads[2]))

        @cuda blocks=blocks threads=threads div_kernel!(div, velocX, velocY, Int32(N))
        set_bnd!(N, 0, div, true); set_bnd!(N, 0, p, true)

        for _ in 1:10
            @cuda blocks=blocks threads=threads pressure_kernel!(p, div, Int32(N))
            set_bnd!(N, 0, p, true)
        end

        @cuda blocks=blocks threads=threads velocity_update_kernel!(velocX, velocY, p, Int32(N))
        set_bnd!(N, 1, velocX, true); set_bnd!(N, 2, velocY, true)
    else
        # CPU version (finite difference / Gauss-Seidel)
        for j in 2:N-1, i in 2:N-1
            div[i,j] = -0.5f0 * ((velocX[i+1,j] - velocX[i-1,j]) + (velocY[i,j+1] - velocY[i,j-1])) / Float32(N)
            p[i,j] = 0.0f0
        end
        set_bnd!(N, 0, div, false); set_bnd!(N, 0, p, false)
        for _ in 1:20
            for j in 2:N-1, i in 2:N-1
                p[i,j] = (div[i,j] + p[i-1,j] + p[i+1,j] + p[i,j-1] + p[i,j+1]) / 4.0f0
            end
            set_bnd!(N, 0, p, false)
        end
        for j in 2:N-1, i in 2:N-1
            velocX[i,j] -= 0.5f0 * Float32(N) * (p[i+1,j] - p[i-1,j])
            velocY[i,j] -= 0.5f0 * Float32(N) * (p[i,j+1] - p[i,j-1])
        end
        set_bnd!(N, 1, velocX, false); set_bnd!(N, 2, velocY, false)
    end
end

function advect!(fluid::FluidSimulation, b, d, d0, velocX, velocY)
    N = fluid.size
    dt0 = fluid.dt * Float32(N)
    if fluid.use_cuda
        function advect_kernel!(d, d0, velocX, velocY, dt0, N)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            if i >= 2 && i <= N-1 && j >= 2 && j <= N-1
                x = clamp(Float32(i) - dt0 * velocX[i, j], Float32(0.5), Float32(N) + Float32(0.5))
                y = clamp(Float32(j) - dt0 * velocY[i, j], Float32(0.5), Float32(N) + Float32(0.5))
                i0 = clamp(floor(Int32, x), Int32(1), Int32(N))
                i1 = clamp(i0 + Int32(1), Int32(1), Int32(N))
                j0 = clamp(floor(Int32, y), Int32(1), Int32(N))
                j1 = clamp(j0 + Int32(1), Int32(1), Int32(N))
                s1 = x - Float32(i0); s0 = Float32(1.0) - s1
                t1 = y - Float32(j0); t0 = Float32(1.0) - t1
                d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
                          s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
            end
            return
        end
        threads = (16, 16)
        blocks = (cld(N, threads[1]), cld(N, threads[2]))
        @cuda blocks=blocks threads=threads advect_kernel!(d, d0, velocX, velocY, Float32(dt0), Int32(N))
        set_bnd!(N, b, d, true)
    else
        for j in 2:N-1, i in 2:N-1
            x = clamp(Float32(i) - dt0 * velocX[i,j], 0.5f0, Float32(N)+0.5f0)
            y = clamp(Float32(j) - dt0 * velocY[i,j], 0.5f0, Float32(N)+0.5f0)
            i0 = clamp(fld(Int(floor(x)),1), 1, N)
            i1 = clamp(i0 + 1, 1, N)
            j0 = clamp(fld(Int(floor(y)),1), 1, N)
            j1 = clamp(j0 + 1, 1, N)
            s1 = x - Float32(i0); s0 = 1.0f0 - s1
            t1 = y - Float32(j0); t0 = 1.0f0 - t1
            d[i,j] = s0*(t0*d0[i0,j0] + t1*d0[i0,j1]) + s1*(t0*d0[i1,j0] + t1*d0[i1,j1])
        end
        set_bnd!(N, b, d, false)
    end
end

# --- apply_slip_boundary! ---
function apply_slip_boundary!(fluid::FluidSimulation)
    N = fluid.size
    if fluid.use_cuda
        function slip_kernel!(Vx, Vy, mask, N)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            if i >= 2 && i <= N-1 && j >= 2 && j <= N-1
                if mask[i, j]
                    mL = Float32(mask[i-1, j])
                    mR = Float32(mask[i+1, j])
                    mD = Float32(mask[i, j-1])
                    mU = Float32(mask[i, j+1])
                    gx = mR - mL
                    gy = mU - mD
                    grad_norm = sqrt(gx*gx + gy*gy)
                    if grad_norm > Float32(1e-6)
                        nx = gx / grad_norm
                        ny = gy / grad_norm
                        wL = Float32(1.0) - mL
                        wR = Float32(1.0) - mR
                        wD = Float32(1.0) - mD
                        wU = Float32(1.0) - mU
                        denom = wL + wR + wD + wU
                        vx_avg = Float32(0.0); vy_avg = Float32(0.0)
                        if denom > Float32(0.0)
                            if wL > Float32(0.0)
                                vx_avg += Vx[i-1, j] * wL; vy_avg += Vy[i-1, j] * wL
                            end
                            if wR > Float32(0.0)
                                vx_avg += Vx[i+1, j] * wR; vy_avg += Vy[i+1, j] * wR
                            end
                            if wD > Float32(0.0)
                                vx_avg += Vx[i, j-1] * wD; vy_avg += Vy[i, j-1] * wD
                            end
                            if wU > Float32(0.0)
                                vx_avg += Vx[i, j+1] * wU; vy_avg += Vy[i, j+1] * wU
                            end
                            vx_avg /= denom; vy_avg /= denom
                        end
                        tx = -ny; ty = nx
                        v_t = vx_avg * tx + vy_avg * ty
                        Vx[i, j] = v_t * tx; Vy[i, j] = v_t * ty
                    else
                        Vx[i, j] = 0.0f0; Vy[i, j] = 0.0f0
                    end
                end
            end
            return
        end
        threads = (16,16)
        blocks = (cld(N, threads[1]), cld(N, threads[2]))
        @cuda blocks=blocks threads=threads slip_kernel!(fluid.Vx, fluid.Vy, fluid.obstacle_mask, Int32(N))
    else
        M = fluid.obstacle_mask
        Vx = fluid.Vx; Vy = fluid.Vy
        for j in 2:N-1, i in 2:N-1
            if M[i,j]
                mL = Float32(M[i-1, j]); mR = Float32(M[i+1, j])
                mD = Float32(M[i, j-1]); mU = Float32(M[i, j+1])
                gx = mR - mL; gy = mU - mD
                grad_norm = sqrt(gx*gx + gy*gy)
                if grad_norm > 1e-6f0
                    nx = gx/grad_norm; ny = gy/grad_norm
                    wL = 1.0f0 - mL; wR = 1.0f0 - mR; wD = 1.0f0 - mD; wU = 1.0f0 - mU
                    denom = wL + wR + wD + wU
                    vx_avg = 0.0f0; vy_avg = 0.0f0
                    if denom > 0.0f0
                        if wL > 0.0f0
                            vx_avg += Vx[i-1,j]*wL; vy_avg += Vy[i-1,j]*wL
                        end
                        if wR > 0.0f0
                            vx_avg += Vx[i+1,j]*wR; vy_avg += Vy[i+1,j]*wR
                        end
                        if wD > 0.0f0
                            vx_avg += Vx[i,j-1]*wD; vy_avg += Vy[i,j-1]*wD
                        end
                        if wU > 0.0f0
                            vx_avg += Vx[i,j+1]*wU; vy_avg += Vy[i,j+1]*wU
                        end
                        vx_avg /= denom; vy_avg /= denom
                    end
                    tx = -ny; ty = nx
                    v_t = vx_avg*tx + vy_avg*ty
                    Vx[i,j] = v_t*tx; Vy[i,j] = v_t*ty
                else
                    Vx[i,j] = 0.0f0; Vy[i,j] = 0.0f0
                end
            end
        end
    end
end

# --- The main step! function that uses the correct backend implementations ---
function step!(fluid::FluidSimulation)
    N = fluid.size
    dx = fluid.dx

    if fluid.mode != "interactive"
        mid_start = floor(Int, N * 0.3)
        mid_end = floor(Int, N * 0.7)
        inflow_width_m = Float32(0.05)
        inflow_width = max(1, round(Int, inflow_width_m / dx))

        if fluid.use_cuda
            fluid.Vx[1:inflow_width, mid_start:mid_end] .= fluid.inflow_velocity
            fluid.Vy[1:inflow_width, mid_start:mid_end] .= 0.0f0
            fluid.density[1:inflow_width, mid_start:mid_end] .= 200.0f0
        else
            fluid.Vx[1:inflow_width, mid_start:mid_end] .= fluid.inflow_velocity
            fluid.Vy[1:inflow_width, mid_start:mid_end] .= 0.0f0
            fluid.density[1:inflow_width, mid_start:mid_end] .= 200.0f0
        end
    end

    # diffusion of velocity if viscosity > 0
    if fluid.visc > 0.0f0
        fluid.Vx0 .= fluid.Vx
        fluid.Vy0 .= fluid.Vy
        diffuse!(fluid, 1, fluid.Vx, fluid.Vx0, fluid.visc)
        diffuse!(fluid, 2, fluid.Vy, fluid.Vy0, fluid.visc)
    end

    project!(fluid, fluid.Vx, fluid.Vy, fluid.Vx0, fluid.Vy0)

    fluid.Vx0 .= fluid.Vx
    fluid.Vy0 .= fluid.Vy
    advect!(fluid, 1, fluid.Vx, fluid.Vx0, fluid.Vx0, fluid.Vy0)
    advect!(fluid, 2, fluid.Vy, fluid.Vy0, fluid.Vx0, fluid.Vy0)

    fluid.s .= fluid.density
    advect!(fluid, 0, fluid.density, fluid.s, fluid.Vx, fluid.Vy)

    project!(fluid, fluid.Vx, fluid.Vy, fluid.Vx0, fluid.Vy0)

    apply_slip_boundary!(fluid)

    zero_where_mask!(fluid, fluid.density)

    fluid.density .*= Float32(0.99)
end

# ---------------------------
# Obstacle generation (same as your code) and wrapper
# ---------------------------

# point_in_polygon as-is
function point_in_polygon(px::Float32, py::Float32, poly::Vector{SVector{2,Float32}})
    inside = false
    n = length(poly)
    j = n
    for i in 1:n
        xi, yi = poly[i][1], poly[i][2]
        xj, yj = poly[j][1], poly[j][2]
        intersect = ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi + Float32(1e-12)) + xi)
        if intersect
            inside = !inside
        end
        j = i
    end
    return inside
end

function set_obstacle!(fluid::FluidSimulation, mode::String, size_param_m::Real; m::Float32=0.02f0, p::Float32=0.4f0, t::Float32=0.12f0, angle::Float32=0.0f0)
    N = fluid.size
    cx_grid = fld(N, 2)
    cy_grid = fld(N, 2)
    dx = fluid.dx

    size_param_m_f = Float32(size_param_m)
    size_grid = max(1, round(Int, size_param_m_f / dx))

    obstacle_mask_cpu = falses(N, N)
    fluid.mode = mode
    fluid.geo_size_m = size_param_m_f

    if mode == "sphere"
        radius = size_grid
        for j in 1:N, i in 1:N
            if ((i - cx_grid)^2 + (j - cy_grid)^2) < radius^2
                obstacle_mask_cpu[i, j] = true
            end
        end

    elseif mode == "aero"
        chord_m = clamp(size_param_m_f * Float32(1.5), Float32(0.02), fluid.domain_size_m * Float32(0.6))
        chord = max(8, round(Int, chord_m / dx))

        m_f = clamp(Float32(m), 0.0f0, 0.1f0)
        p_f = clamp(Float32(p), 0.05f0, 0.95f0)
        t_f = clamp(Float32(t), 0.02f0, 0.30f0)
        angle_f = Float32(angle)

        x_le_grid = Int(round(cx_grid - chord/2))
        num = max(200, chord * 3)
        x_rel = range(0.0f0, 1.0f0, length = num)

        yt = Float32.(5.0f0) .* t_f .* (0.2969f0 .* sqrt.(x_rel) .- 0.1260f0 .* x_rel .- 0.3516f0 .* (x_rel.^2) .+
                                         0.2843f0 .* (x_rel.^3) .- 0.1015f0 .* (x_rel.^4))
        yc = zeros(Float32, length(x_rel))
        dyc_dx = zeros(Float32, length(x_rel))
        if p_f > 0.0f0 && m_f > 0.0f0
            for idx in eachindex(x_rel)
                xval = x_rel[idx]
                if xval < p_f
                    yc[idx] = (m_f / p_f^2) * (2f0 * p_f * xval - xval^2)
                    dyc_dx[idx] = (2f0*m_f / p_f^2) * (p_f - xval)
                else
                    yc[idx] = (m_f / (1f0 - p_f)^2) * ((1f0 - 2f0*p_f) + 2f0*p_f*xval - xval^2)
                    dyc_dx[idx] = (2f0*m_f / (1f0 - p_f)^2) * (p_f - xval)
                end
            end
        end

        x_abs = Float32.(x_le_grid) .+ Float32.(x_rel) .* Float32(chord)
        yc_abs = Float32(cy_grid) .+ yc .* Float32(chord)
        yt_abs = yt .* Float32(chord)
        theta = atan.(dyc_dx)

        xu = x_abs .- yt_abs .* sin.(theta)
        yu = yc_abs .+ yt_abs .* cos.(theta)
        xl = x_abs .+ yt_abs .* sin.(theta)
        yl = yc_abs .- yt_abs .* cos.(theta)

        polygon = Vector{SVector{2,Float32}}()
        for k in 1:length(xu)
            push!(polygon, SVector{2,Float32}(xu[k], yu[k]))
        end
        for k in length(xl):-1:1
            push!(polygon, SVector{2,Float32}(xl[k], yl[k]))
        end

        angle_rad = angle_f * (pi/180f0)
        cosA = cos(angle_rad); sinA = sin(angle_rad)
        for idx in 1:length(polygon)
            v = polygon[idx] .- SVector{2,Float32}(Float32(cx_grid), Float32(cy_grid))
            rotated = SVector{2,Float32}(cosA * v[1] - sinA * v[2], sinA * v[1] + cosA * v[2]) .+ SVector{2,Float32}(Float32(cx_grid), Float32(cy_grid))
            polygon[idx] = rotated
        end

        minx = clamp(Int(floor(minimum(map(v->v[1], polygon)))), 1, N)
        maxx = clamp(Int(ceil(maximum(map(v->v[1], polygon)))), 1, N)
        miny = clamp(Int(floor(minimum(map(v->v[2], polygon)))), 1, N)
        maxy = clamp(Int(ceil(maximum(map(v->v[2], polygon)))), 1, N)

        for j in miny:maxy, i in minx:maxx
            if point_in_polygon(Float32(i), Float32(j), polygon)
                obstacle_mask_cpu[i, j] = true
            end
        end

    elseif mode == "triangles"
        L_grid = clamp(size_grid, 8, round(Int, Float32(N) * Float32(0.45)))
        tri_width = L_grid * 2
        tri_height = clamp(size_grid, 8, round(Int, Float32(N) * Float32(0.45)))
        half_len = tri_width ÷ 2
        half_h = tri_height ÷ 2
        x1 = clamp(cx_grid - half_len, 1, N); y1 = clamp(cy_grid, 1, N)
        x2 = clamp(cx_grid + half_len, 1, N); y2 = clamp(cy_grid - half_h, 1, N)
        x3 = clamp(cx_grid + half_len, 1, N); y3 = clamp(cy_grid + half_h, 1, N)
        denom = Float32((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
        if abs(denom) > Float32(1e-8)
            minx = max(1, min(x1, min(x2, x3)) - 1)
            maxx = min(N, max(x1, max(x2, x3)) + 1)
            miny = max(1, min(y1, min(y2, y3)) - 1)
            maxy = min(N, max(y1, max(y2, y3)) + 1)
            for j in miny:maxy, i in minx:maxx
                a = Float32(((y2 - y3)*(i - x3) + (x3 - x2)*(j - y3))) / denom
                b = Float32(((y3 - y1)*(i - x3) + (x1 - x3)*(j - y3))) / denom
                c = 1.0f0 - a - b
                if a >= 0f0 && b >= 0f0 && c >= 0f0
                    obstacle_mask_cpu[i, j] = true
                end
            end
        end
    end

    upload_mask!(fluid, obstacle_mask_cpu)

    if any(obstacle_mask_cpu)
        zero_where_mask!(fluid, fluid.density)
        zero_where_mask!(fluid, fluid.Vx)
        zero_where_mask!(fluid, fluid.Vy)
    end

    return obstacle_mask_cpu
end

# ---------------------------
# GUI and main (identical logic; uses fluid.use_cuda flag)
# ---------------------------
function main()
    # choose sizes based on availability
    use_cuda = HAS_CUDA
    sim_size = use_cuda ? 1024 : 512
    domain_size_m = 1.0

    fluid = make_simulation(use_cuda=use_cuda, gpu_size=1024, cpu_size=512, domain_size_gpu=domain_size_m)

    fig = Figure(size = (1400, 900), fontsize = 18)
    fig[1, 1] = Label(fig[1, 1], "Fluid Dynamics — mode: $(fluid.use_cuda ? "GPU (CUDA)" : "CPU fallback") — grid: $(fluid.size)×$(fluid.size)", fontsize = 20)

    # Controls
    fig[2, 1] = Label(fig[2, 1], "Inflow Velocity (m/s)", fontsize = 14)
    sl_vel = Slider(fig[3, 1], range = 1.0:0.1:10.0, startvalue = 0.0)
    fig[3, 1] = sl_vel

    fig[4, 1] = Label(fig[4, 1], "Kinematic Viscosity ν (m²/s) — max = water", fontsize = 14)
    step_visc = 1e-7
    sl_visc = Slider(fig[5, 1], range = 0.0:step_visc:NU_WATER, startvalue = NU_WATER)
    fig[5, 1] = sl_visc

    fig[6, 1] = Label(fig[6, 1], lift(sl_visc.value) do v "Current ν: $(v) m²/s" end; fontsize = 12)

    fig[7, 1] = Label(fig[7, 1], "Geometry size (meters)", fontsize = 14)
    sl_size = Slider(fig[8, 1], range = 0.001:0.001:0.5, startvalue = fluid.geo_size_m)
    fig[8, 1] = sl_size

    # Aero sliders
    fig[9, 1] = Label(fig[9, 1], "Airfoil Parameters", fontsize = 14)
    fig[10, 1] = Label(fig[10, 1], "Thickness t (fraction)", fontsize = 12)
    sl_t = Slider(fig[11, 1], range = 0.02:0.01:0.25, startvalue = 0.12)
    fig[11, 1] = sl_t

    fig[12, 1] = Label(fig[12, 1], "Camber m (fraction)", fontsize = 12)
    sl_m = Slider(fig[13, 1], range = 0.0:0.005:0.1, startvalue = 0.02)
    fig[13, 1] = sl_m

    fig[14, 1] = Label(fig[14, 1], "Camber pos p (fraction)", fontsize = 12)
    sl_p = Slider(fig[15, 1], range = 0.1:0.05:0.9, startvalue = 0.4)
    fig[15, 1] = sl_p

    fig[16, 1] = Label(fig[16, 1], "Angle (deg)", fontsize = 12)
    sl_angle = Slider(fig[17, 1], range = -20:1:20, startvalue = 0.0)
    fig[17, 1] = sl_angle

    fig[18, 1] = Label(fig[18, 1], "Simulation Modes", fontsize = 14)

    btn_inter = Button(fig[19, 1]; label = "1. Interactive")
    fig[19, 1] = btn_inter
    btn_sphere = Button(fig[20, 1]; label = "2. Sphere Flow")
    fig[20, 1] = btn_sphere
    btn_aero = Button(fig[21, 1]; label = "3. Aerodynamic")
    fig[21, 1] = btn_aero
    btn_tri = Button(fig[22, 1]; label = "4. Triangles")
    fig[22, 1] = btn_tri
    btn_clear = Button(fig[23, 1]; label = "Clear Fluid")
    fig[23, 1] = btn_clear

    ax = Axis(fig[1:23, 2], title = "Density — physical domain: $(round(fluid.domain_size_m, sigdigits=4)) m × $(round(fluid.domain_size_m, sigdigits=4)) m",
              aspect = DataAspect())
    hidedecorations!(ax)

    try
        deregister_interaction!(ax, :rectanglezoom)
        deregister_interaction!(ax, :scrollzoom)
    catch e
        @warn "Couldn't deregister some interactions: $e"
    end

    density_node = Observable(zeros(Float32, sim_size, sim_size))
    obstacle_node = Observable(zeros(Float32, sim_size, sim_size))

    hm = heatmap!(ax, density_node, colormap = :turbo, colorrange = (0, 255))
    hm_obs = heatmap!(ax, obstacle_node, colormap = [RGBAf(0,0,0,0), RGBAf(1,0,0,0.45)], colorrange = (0, 1), interpolate = false)
    cbar = Colorbar(fig[1:23, 3], hm; label = "Density", height = Relative(1.0))

    mouse_active = Observable(false)
    prev_mouse_pos = Observable(Vec2f(0.0, 0.0))

    on(btn_inter.clicks) do _
        fluid.mode = "interactive"
        sl_vel.value[] = 0.0
        if fluid.use_cuda
            fluid.obstacle_mask .= false
        else
            fluid.obstacle_mask .= falses(fluid.size, fluid.size)
        end
        obstacle_node[] .= 0.0f0
    end

    on(btn_sphere.clicks) do _
        fluid.mode = "sphere"
        sl_vel.value[] = 1.0
        mask_cpu = set_obstacle!(fluid, "sphere", sl_size.value[])
        obstacle_node[] = Float32.(mask_cpu)
    end

    on(btn_aero.clicks) do _
        fluid.mode = "aero"
        sl_vel.value[] = 1.0
        mask_cpu = set_obstacle!(fluid, "aero", sl_size.value[]; m=Float32(sl_m.value[]), p=Float32(sl_p.value[]), t=Float32(sl_t.value[]), angle=Float32(sl_angle.value[]))
        obstacle_node[] = Float32.(mask_cpu)
    end

    on(btn_tri.clicks) do _
        fluid.mode = "triangles"
        sl_vel.value[] = 1.0
        mask_cpu = set_obstacle!(fluid, "triangles", sl_size.value[])
        obstacle_node[] = Float32.(mask_cpu)
    end

    on(btn_clear.clicks) do _
        fluid.density .= Float32(0.0)
        fluid.Vx .= Float32(0.0)
        fluid.Vy .= Float32(0.0)
        if fluid.use_cuda
            fluid.obstacle_mask .= false
        else
            fluid.obstacle_mask .= falses(fluid.size, fluid.size)
        end
        obstacle_node[] .= 0.0f0
    end

    # update geometry when sliders change (same logic)
    on(sl_size.value) do val
        if fluid.mode != "interactive"
            if fluid.mode == "aero"
                mask_cpu = set_obstacle!(fluid, fluid.mode, val; m=Float32(sl_m.value[]), p=Float32(sl_p.value[]), t=Float32(sl_t.value[]), angle=Float32(sl_angle.value[]))
            else
                mask_cpu = set_obstacle!(fluid, fluid.mode, val)
            end
            obstacle_node[] = Float32.(mask_cpu)
        end
    end
    for obs in (sl_m, sl_p, sl_t, sl_angle)
        on(obs.value) do _
            if fluid.mode == "aero"
                mask_cpu = set_obstacle!(fluid, "aero", sl_size.value[]; m=Float32(sl_m.value[]), p=Float32(sl_p.value[]), t=Float32(sl_t.value[]), angle=Float32(sl_angle.value[]))
                obstacle_node[] = Float32.(mask_cpu)
            end
        end
    end

    on(events(ax.scene).mousebutton) do event
        if event.button == Mouse.left
            if event.action == Mouse.press
                mouse_active[] = true
                pos = mouseposition(ax.scene)
                if all(isfinite, pos)
                    prev_mouse_pos[] = pos
                end
            elseif event.action == Mouse.release
                mouse_active[] = false
            end
        end
    end

    is_running = Observable(true)
    frame_counter = 0
    last_update_time = time()

    # Main loop (uses CPU/CPU-GPU branches internally)
    @async while is_running[]
        if fluid.use_cuda
            CUDA.device!(0)
        end

        fluid.inflow_velocity = Float32(sl_vel.value[])
        fluid.visc = Float32(sl_visc.value[])

        if mouse_active[]
            pos = mouseposition(ax.scene)
            if all(isfinite, pos)
                mx, my = pos[1], pos[2]
                ix = clamp(round(Int, mx), 1, sim_size)
                iy = clamp(round(Int, my), 1, sim_size)
                prev_pos = prev_mouse_pos[]
                px, py = prev_pos[1], prev_pos[2]
                add_density!(fluid, ix, iy, 200.0)
                force_x = Float32((mx - px) * 10.0)
                force_y = Float32((my - py) * 10.0)
                add_velocity!(fluid, ix, iy, force_x, force_y)
                prev_mouse_pos[] = pos
            end
        end

        step!(fluid)
        frame_counter += 1
        current_time = time()
        if current_time - last_update_time >= 0.033
            # copy density & obstacle to CPU for plotting
            if fluid.use_cuda
                density_cpu = Array(fluid.density)
                obstacle_cpu = Array(fluid.obstacle_mask)
            else
                density_cpu = fluid.density
                obstacle_cpu = fluid.obstacle_mask
            end
            density_node[] = Float32.(density_cpu)
            obstacle_node[] = Float32.(obstacle_cpu)
            last_update_time = current_time
        end
        sleep(0.001)
    end

    on(events(fig).window_open) do open
        if !open
            is_running[] = false
        end
    end

    display(fig)
    return fig
end

# Run main
main()
