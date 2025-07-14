import numpy as np
import matplotlib.pyplot as plt



# -----------------------------
# Utility functions
# -----------------------------

def initialize_domain(lx, ly, nx, ny):
    x=np.linspace(0., lx, num=nx, endpoint=False)
    y=np.linspace(0., ly, num=ny, endpoint=False)
    X, Y=np.meshgrid(x, y, indexing='ij')
    dkx=2.*np.pi/lx
    dky=2.*np.pi/ly
    kx=np.fft.ifftshift(dkx*np.arange(-nx/2, nx/2))
    ky=np.fft.ifftshift(dky*np.arange(-ny/2, ny/2))
    KX, KY=np.meshgrid(kx, ky, indexing='ij')
    return x, y, X, Y, kx, ky, KX, KY


def find_guesses(psi, density_threshold, x, y, extraMask=None):
    mask=(np.abs(psi)**2<=density_threshold)
    if extraMask is not None:
        mask=mask & extraMask
    indices=np.argwhere(mask)
    return np.array([[x[i], y[j]] for i, j in indices])


def shift_at_position(psi, KX, KY, r0):
    x0, y0=r0
    return np.fft.ifftn(np.fft.fftn(psi)*np.exp(1j*(KX*x0+KY*y0)))


def Jacobian_at_origin(psi, KX, KY):
    psitilde=np.fft.fftn(psi)
    psi_x=1j*np.fft.ifftn(psitilde*KX)
    psi_y=1j*np.fft.ifftn(psitilde*KY)
    return np.array([
        [np.real(psi_x[0, 0]), np.real(psi_y[0, 0])],
        [np.imag(psi_x[0, 0]), np.imag(psi_y[0, 0])]
    ])


def pseudovorticity_at_origin(psi, KX, KY):
    psitilde=np.fft.fftn(psi)
    psi_x=1j*np.fft.ifftn(psitilde*KX)
    psi_y=1j*np.fft.ifftn(psitilde*KY)
    return np.real(psi_x[0, 0])*np.imag(psi_y[0, 0])-np.real(psi_y[0, 0])*np.imag(psi_x[0, 0])


def run_Newton(psi, KX, KY, r0, nstepNewton, tolerance):
    psi_temp=shift_at_position(psi, KX, KY, r0)
    for _ in range(nstepNewton):
        density=np.abs(psi_temp[0, 0])
        #print("density = ", density, ", r0 = ", r0)
        if (density < tolerance):
            return r0, 0
        J=Jacobian_at_origin(psi_temp, KX, KY)
        delta_r0=np.linalg.solve(J, [np.real(psi_temp[0, 0]), np.imag(psi_temp[0, 0])])
        r0=r0-delta_r0
        psi_temp=shift_at_position(psi_temp, KX, KY, -delta_r0)
    return r0, 1


def circulation_at_origin(psi, KX, KY, epsilon=0.1, nPoints=32, ax=None):
    angVel=np.zeros(nPoints, dtype=np.float64)
    thetas=np.linspace(0, 2*np.pi, nPoints, endpoint=False)
    psiTemp=np.zeros(nPoints, dtype=np.complex128)

    for i in range(0, nPoints):
        r0=epsilon*np.cos(thetas[i]), epsilon*np.sin(thetas[i])
        psiTemp[i]=shift_at_position(psi, KX, KY, r0)[0, 0]

    kTheta=np.fft.ifftshift(np.arange(-nPoints/2, nPoints/2))
    psi_theta=1j*np.fft.ifft(kTheta*np.fft.fft(psiTemp))
    
    angVel=np.imag(np.conjugate(psiTemp)*psi_theta)/(np.abs(psiTemp)**2)

    circ=np.real(np.fft.fft(angVel)[0])/nPoints

    if ax is None:
        return circ

    else:
        ax.plot(thetas, angVel, 'o')
        ax.set_xlabel("theta")
        ax.set_ylabel("angular velocity")
        return circ, ax
    


# -----------------------------
# Main script
# -----------------------------

if __name__ == "__main__":
    
    # Set parameters (edit these values if needed!)
    lx = 128.                   # domain size along x-axis
    ly = 128.                   # domain size along y-axis
    nx = 256                    # number of grid points along x-axis
    ny = 256                    # number of grid points along x-axis
    
    density_threshold = 1e-01   # density value below which identifying a vortex guess 
    tolerance = 1e-10           # density threshold below which identifying a vortex 
    nstepNewton = 10            # maximum number of Newton iteration when searching a vortex
    minVortDistance = 0.2       # minimum distance at which identifying two different vortices 
    epsilonCirc = 0.01          # radius to compute the circulation
    nCircPoints = 16            # number of points used to compute the circulation

    print("Parameters set")


    # Initialise domain grid and operators
    x, y, X, Y, kx, ky, KX, KY=initialize_domain(lx, ly, nx, ny)

    print("Domain grid and operators initialised")


    # Load complex field psi from file (edit this if needed!)
    fname="./psiField2D.npy"
    psi=np.load(fname)          # filename of the field psi
    print(fname)

    print("Complex field psi loaded")


    # Plot density and phase fields before tracking
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.3])

    ax1 = fig.add_subplot(gs[0, 0])
    pcm1 = ax1.pcolormesh(x, y, (np.abs(psi)**2).T, shading='auto')
    cbar = fig.colorbar(pcm1, ax=ax1, orientation='horizontal', pad=0.15, fraction=0.046)
    cbar.set_label('Density color scale')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Density")
    ax1.set_aspect('equal')

    ax2 = fig.add_subplot(gs[0, 1])
    pcm2 = ax2.pcolormesh(x, y, np.angle(psi).T, shading='auto')
    cbar = fig.colorbar(pcm2, ax=ax2, orientation='horizontal', pad=0.15, fraction=0.046)
    cbar.set_label('Phase color scale')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Phase")
    ax2.set_aspect('equal')

    plt.tight_layout()
    fname="./fields.png"
    plt.savefig(fname)
    print(fname)
    plt.show()

    print("Density and phase fields before tracking plotted")


    # Adding extra radial mask (edit these values if needed!)
    #R=30.
    #x0=lx/2.
    #y0=ly/2.
    #extraMask=((X-x0)**2+(Y-y0)**2<R**2)

    #print("Extra radial mask added")


    # Find vortex position guesses
    guesses=find_guesses(psi, density_threshold, x, y)
    #guesses=find_guesses(psi, density_threshold, x, y, extraMask=extraMask)

    print("Vortex position guesses found")


    # Execute Newton iteration to find vortices
    vortexPoints=[]
    pseudovorticity=[]
    circulation=[]
    
    for r0 in guesses:
        r0_new, check=run_Newton(psi, KX, KY, r0, nstepNewton, tolerance)
        if check == 0:
            flag = True
            if vortexPoints:
                for vortex in vortexPoints:
                    if np.linalg.norm(vortex-r0_new) < minVortDistance:
                        flag = False
            if flag:
                vortexPoints.append(r0_new)
                pseudovorticity.append(
                    pseudovorticity_at_origin(shift_at_position(psi, KX, KY, r0_new), KX, KY))
                circulation.append(
                    circulation_at_origin(shift_at_position(psi, KX, KY, r0_new), KX, KY, epsilon=epsilonCirc, nPoints=nCircPoints)
                )
        else:
            print("Convergence for guess ", r0, " not reached!")

    vortexPoints=np.array([vortexPoints])[0, :, :]
    vortNum=vortexPoints.shape[0]
    pseudovorticity=np.array([pseudovorticity])[0, :]
    circulation=np.array([circulation])[0, :]

    print("Newton iteration to find vortices executed")


    # Output results
    if (vortNum>0):
        print(vortNum, "vortex points detected")
        print("x-coordinate, y-coordinate, pseudovorticity, circulation")
        for i in range(0, vortNum):
            print(vortexPoints[i, 0], vortexPoints[i, 1], pseudovorticity[i], circulation[i])
    else:
        print("No vortex points were detected")

    # Save result
    data=np.column_stack((vortexPoints[:, 0], vortexPoints[:, 1], pseudovorticity))
    fname="detectedVortices.txt"
    np.savetxt(fname, data, delimiter=' ', header='x-coordinate, y-coordinate, pseudovorticity, circulation')
    print(fname)

    print("Results output")


    # Plot density and phase fields after tracking
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.3])

    ax1 = fig.add_subplot(gs[0, 0])
    pcm1 = ax1.pcolormesh(x, y, (np.abs(psi)**2).T, shading='auto')
    if (vortNum>0):
        mask=np.where(pseudovorticity>0)
        ax1.scatter(vortexPoints[:, 0][mask], vortexPoints[:, 1][mask], color='red', s=20, label='positive circulation')
        mask=np.where(pseudovorticity<0)
        ax1.scatter(vortexPoints[:, 0][mask], vortexPoints[:, 1][mask], color='blue', s=20, label='negative circulation')
    cbar = fig.colorbar(pcm1, ax=ax1, orientation='horizontal', pad=0.15, fraction=0.046)
    cbar.set_label('Density color scale')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Density")
    ax1.set_aspect('equal')

    ax2 = fig.add_subplot(gs[0, 1])
    pcm2 = ax2.pcolormesh(x, y, np.angle(psi).T, shading='auto')
    if (vortNum>0):
        mask=np.where(pseudovorticity>0)
        ax2.scatter(vortexPoints[:, 0][mask], vortexPoints[:, 1][mask], color='red', s=20, label='positive circulation')
        mask=np.where(pseudovorticity<0)
        ax2.scatter(vortexPoints[:, 0][mask], vortexPoints[:, 1][mask], color='blue', s=20, label='negative circulation')
    cbar = fig.colorbar(pcm2, ax=ax2, orientation='horizontal', pad=0.15, fraction=0.046)
    cbar.set_label('Phase color scale')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Phase")
    ax2.set_aspect('equal')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    scatter_labels = ['positive', 'negative']
    scatter_colors = ['red', 'blue']
    scatter_markers = ['o', 'o']
    handles = [
        plt.Line2D([], [], marker=marker, color='w', label=label,
                markerfacecolor=color, markersize=10)
        for marker, color, label in zip(scatter_markers, scatter_colors, scatter_labels)
    ]
    title = f"{vortNum} vortices detected\nof circulation"
    ax3.legend(handles=handles, loc='center', title=title, frameon=False)

    plt.tight_layout()
    fname="./fieldsWithVortices.png"
    plt.savefig(fname)
    print(fname)
    plt.show()

    print("Density and phase fields after tracking plotted")


    # Zoom into a vortex point, estimated angular velocity and circulation (edit these values if needed!)
    nCircPoints=64
    epsilonCirc=0.1
    nxZoom=32
    nyZoom=32
    vortexIndex=10

    r0=vortexPoints[vortexIndex, 0], vortexPoints[vortexIndex, 1]
    xZoom=np.linspace(r0[0]-1.5*epsilonCirc, r0[0]+1.5*epsilonCirc, num=nxZoom, endpoint=False)
    yZoom=np.linspace(r0[1]-1.5*epsilonCirc, r0[1]+1.5*epsilonCirc, num=nxZoom, endpoint=False)

    circPoints=np.empty((nCircPoints, 2), dtype=np.float64)
    for i in range(0, nCircPoints):
        circPoints[i, 0]=r0[0]+epsilonCirc*np.cos(2*np.pi/nCircPoints*i)
        circPoints[i, 1]=r0[1]+epsilonCirc*np.sin(2*np.pi/nCircPoints*i)

    psiZoom=np.empty((nxZoom, nyZoom), dtype=np.complex128)
    for i in range(0, nxZoom):
        for j in range(0, nyZoom):
            r0_new=xZoom[i], yZoom[j]
            psiZoom[i, j]=shift_at_position(psi, KX, KY, r0_new)[0, 0]

    fig = plt.figure(figsize=(17, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.3])

    ax1 = fig.add_subplot(gs[0, 0])
    pcm1 = ax1.pcolormesh(xZoom, yZoom, (np.abs(psiZoom)**2).T, shading='auto')
    ax1.scatter(circPoints[:, 0], circPoints[:, 1], color='black', s=20, label='angular velocity points')
    cbar = fig.colorbar(pcm1, ax=ax1, orientation='horizontal', pad=0.15, fraction=0.046)
    cbar.set_label('Density color scale')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Zoomed density")
    ax1.set_aspect('equal')

    ax2 = fig.add_subplot(gs[0, 1])
    pcm2 = ax2.pcolormesh(xZoom, yZoom, np.angle(psiZoom).T, shading='auto')
    ax2.scatter(circPoints[:, 0], circPoints[:, 1], color='black', s=20, label='angular velocity points')
    cbar = fig.colorbar(pcm2, ax=ax2, orientation='horizontal', pad=0.15, fraction=0.046)
    cbar.set_label('Phase color scale')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Zoomed phase")
    ax2.set_aspect('equal')

    ax3 = fig.add_subplot(gs[0, 2])
    circ, ax3=circulation_at_origin(shift_at_position(psi, KX, KY, r0), KX, KY, epsilon=0.005, nPoints=nCircPoints, ax=ax3)
    ax3.set_title("Estimated angular velocity about the vortex")

    ax4 = fig.add_subplot(gs[0, 3])
    text = (
        f"\nVortex point at\n({r0[0]:.5f}, {r0[1]:.5f}),\nwith measured circulation equal to {circ}."
        f"\nHere the angular velocity is estimated using {nCircPoints} points equally spaced on a circle having radius {epsilonCirc:.3f}."
    )
    ax4.text(-0.3, 0.7, text, fontsize=12, va='top', ha='left', wrap=True)
    ax4.axis('off')

    plt.tight_layout()
    fname="./exampleZoomAndCirculation.png"
    plt.savefig(fname)
    print(fname)
    plt.show()

    print("Zoomed density and phase fields, and angular velocity plotted")


    # Acknowledgments and reference
    print("\nCode released by Davide Proment, MIT License, last update 14 July 2025")
    print("Based on the ideas and results presented in")
    print("Alberto Villois et al. 2016 J. Phys. A: Math. Theor. 49 415502")
    print("https://doi.org/10.1088/1751-8113/49/41/415502\n")
